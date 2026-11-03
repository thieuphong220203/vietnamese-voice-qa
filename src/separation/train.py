"""Fine-tune SepFormer trên dữ liệu trộn tiếng Việt (VIVOS).

Bắt đầu từ model tiếng Anh ``speechbrain/sepformer-whamr16k`` rồi train tiếp
bằng hàm mất mát -SI-SNR có PIT (giống công thức gốc của SpeechBrain).

Lưu trong ``--output-dir``:
    last.pt       checkpoint mới nhất (có optimizer, để chạy tiếp khi Colab ngắt)
    best.pt       checkpoint có SI-SNRi trên tập valid cao nhất (dùng để đánh giá/pipeline)
    train_log.csv loss + SI-SNRi theo từng epoch

Ví dụ (Colab, GPU T4)::

    python -m src.separation.train \\
        --train-csv data/processed/separation/train.csv \\
        --valid-csv data/processed/separation/valid.csv \\
        --output-dir /content/drive/MyDrive/vvqa/checkpoints/separation/finetuned \\
        --epochs 5 --segment-seconds 3 --batch-size 1 --grad-accum 4 --fp16

    # Chạy tiếp sau khi bị ngắt:
    python -m src.separation.train ... --resume /content/drive/.../last.pt

Hết bộ nhớ GPU (CUDA out of memory) → giảm ``--segment-seconds`` (2 hoặc 1.5).
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data.wav_utils import read_wav
from src.separation.metrics import si_snr_improvement
from src.separation.sepformer import DEFAULT_MODEL_ID, load_sepformer, resolve_device

SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- dữ liệu
def read_rows(csv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit else rows


def load_example(
    row: dict[str, str], base_dir: Path, segment_samples: int | None, rng: random.Random | None
) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (mix [T], sources [T, 2])."""

    signals = [read_wav(base_dir / row[key])[0] for key in ("mix", "s1", "s2")]
    length = min(len(signal) for signal in signals)
    signals = [signal[:length] for signal in signals]
    if segment_samples and length > segment_samples:
        start = rng.randint(0, length - segment_samples) if rng else 0
        signals = [signal[start : start + segment_samples] for signal in signals]
    mix, s1, s2 = signals
    return mix, np.stack([s1, s2], axis=-1)


def make_dataset(csv_path: Path, segment_seconds: float, seed: int):
    import torch

    rows = read_rows(csv_path)
    base_dir = csv_path.parent
    segment = round(segment_seconds * SAMPLE_RATE) if segment_seconds > 0 else None

    class MixtureDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            rng = random.Random(seed * 100_003 + index + int(time.time()))
            mix, sources = load_example(rows[index], base_dir, segment, rng)
            return torch.from_numpy(mix), torch.from_numpy(sources)

    return MixtureDataset()


def collate(batch):
    import torch

    length = min(mix.shape[0] for mix, _ in batch)  # cắt về mẫu ngắn nhất trong batch
    mixes = torch.stack([mix[:length] for mix, _ in batch])
    sources = torch.stack([src[:length] for _, src in batch])
    return mixes, sources


# --------------------------------------------------------------------------- model
def forward_separation(model: Any, mix):
    """Giống ``SepformerSeparation.separate_batch`` nhưng giữ gradient. mix [B, T] -> [B, T, S]."""

    import torch
    import torch.nn.functional as F

    mods = model.mods
    num_spks = int(model.hparams.num_spks)
    mix_w = mods.encoder(mix)
    est_mask = mods.masknet(mix_w)
    mix_w = torch.stack([mix_w] * num_spks)
    sep_h = mix_w * est_mask
    est = torch.cat([mods.decoder(sep_h[i]).unsqueeze(-1) for i in range(num_spks)], dim=-1)

    target_len, est_len = mix.shape[1], est.shape[1]
    if est_len < target_len:
        est = F.pad(est, (0, 0, 0, target_len - est_len))
    return est[:, :target_len, :]


def si_snr_torch(estimate, target, eps: float = 1e-8):
    """estimate/target [B, T] -> SI-SNR [B] (dB)."""

    import torch

    estimate = estimate - estimate.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    dot = torch.sum(estimate * target, dim=1, keepdim=True)
    projection = dot * target / (torch.sum(target**2, dim=1, keepdim=True) + eps)
    noise = estimate - projection
    ratio = torch.sum(projection**2, dim=1) / (torch.sum(noise**2, dim=1) + eps)
    return 10 * torch.log10(ratio + eps)


def pit_si_snr_loss(estimates, targets):
    """-SI-SNR với hoán vị tốt nhất cho từng mẫu. estimates/targets [B, T, S]."""

    import itertools

    import torch

    num_spks = targets.shape[-1]
    per_perm = []
    for perm in itertools.permutations(range(num_spks)):
        score = sum(si_snr_torch(estimates[:, :, p], targets[:, :, i]) for i, p in enumerate(perm))
        per_perm.append(score / num_spks)
    best = torch.stack(per_perm, dim=1).max(dim=1).values
    return -best.mean()


def trainable_state(model: Any) -> dict[str, Any]:
    return {name: getattr(model.mods, name).state_dict() for name in ("encoder", "masknet", "decoder")}


def set_trainable(model: Any, train: bool) -> None:
    for name in ("encoder", "masknet", "decoder"):
        module = getattr(model.mods, name)
        module.train(train)
        for parameter in module.parameters():
            parameter.requires_grad_(True)


# --------------------------------------------------------------------------- vòng train
def validate(model: Any, csv_path: Path, device: str, limit: int) -> float:
    import torch

    rows = read_rows(csv_path, limit)
    set_trainable(model, False)
    scores = []
    with torch.no_grad():
        for row in rows:
            mix, sources = load_example(row, csv_path.parent, None, None)
            est = forward_separation(model, torch.from_numpy(mix).unsqueeze(0).to(device))
            est = est[0].float().cpu().numpy()
            result = si_snr_improvement([est[:, 0], est[:, 1]], [sources[:, 0], sources[:, 1]], mix)
            scores.append(result["si_snri"])
    set_trainable(model, True)
    return float(np.mean(scores)) if scores else float("nan")


def train(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.fp16 and device.startswith("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_sepformer(args.model_id, None, device)
    if args.init_checkpoint:
        from src.separation.sepformer import load_finetuned_weights

        load_finetuned_weights(model, args.init_checkpoint, device)
    set_trainable(model, True)
    parameters = [p for name in ("encoder", "masknet", "decoder") for p in getattr(model.mods, name).parameters()]

    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_score, global_step = 1, float("-inf"), 0

    if args.resume:
        state = torch.load(str(args.resume), map_location=device, weights_only=False)
        for name in ("encoder", "masknet", "decoder"):
            getattr(model.mods, name).load_state_dict(state[name])
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + (0 if state.get("mid_epoch") else 1)
        best_score = float(state.get("best_score", best_score))
        global_step = int(state.get("global_step", 0))
        print(f"Chạy tiếp từ {args.resume} (epoch {start_epoch}, best SI-SNRi={best_score:.2f})")

    loader = torch.utils.data.DataLoader(
        make_dataset(args.train_csv, args.segment_seconds, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    if start_epoch == 1 and not args.resume and args.valid_csv:
        baseline = validate(model, args.valid_csv, device, args.valid_limit)
        print(f"SI-SNRi valid trước fine-tune: {baseline:.2f} dB")
        best_score = max(best_score, baseline) if args.keep_baseline_as_best else best_score
        append_log(args.output_dir, {"epoch": 0, "train_loss": "", "valid_si_snri": baseline, "lr": args.lr})

    def save(path: Path, epoch: int, mid_epoch: bool) -> None:
        torch.save(
            {
                **trainable_state(model),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "mid_epoch": mid_epoch,
                "global_step": global_step,
                "best_score": best_score,
                "model_id": args.model_id,
            },
            str(path),
        )

    for epoch in range(start_epoch, args.epochs + 1):
        losses = []
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        for step, (mix, targets) in enumerate(loader, start=1):
            mix, targets = mix.to(device), targets.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                estimates = forward_separation(model, mix)
            loss = pit_si_snr_loss(estimates.float(), targets.float())

            if not torch.isfinite(loss):
                print(f"  bỏ qua batch {step}: loss không hợp lệ")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, args.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.save_every and global_step % args.save_every == 0:
                    save(args.output_dir / "last.pt", epoch, mid_epoch=True)

            losses.append(float(loss.item()))
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss={np.mean(losses[-args.log_every:]):.3f}")
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break

        train_loss = float(np.mean(losses)) if losses else float("nan")
        score = validate(model, args.valid_csv, device, args.valid_limit) if args.valid_csv else -train_loss
        print(
            f"Epoch {epoch}: train loss={train_loss:.3f} | valid SI-SNRi={score:.2f} dB "
            f"| {time.perf_counter() - started:.0f}s"
        )
        append_log(args.output_dir, {"epoch": epoch, "train_loss": train_loss, "valid_si_snri": score, "lr": args.lr})

        if score > best_score:
            best_score = score
            torch.save({**trainable_state(model), "epoch": epoch, "valid_si_snri": score,
                        "model_id": args.model_id}, str(args.output_dir / "best.pt"))
            print(f"  → lưu best.pt (SI-SNRi={score:.2f} dB)")
        save(args.output_dir / "last.pt", epoch, mid_epoch=False)

    print(f"Xong. Best valid SI-SNRi = {best_score:.2f} dB. Checkpoint: {args.output_dir}")


def append_log(output_dir: Path, row: dict[str, Any]) -> None:
    path = output_dir / "train_log.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "valid_si_snri", "lr"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/separation/train.csv"))
    parser.add_argument("--valid-csv", type=Path, default=Path("data/processed/separation/valid.csv"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--init-checkpoint", default=None, help="bắt đầu từ trọng số .pt khác thay vì model gốc")
    parser.add_argument("--resume", type=Path, default=None, help="last.pt để chạy tiếp")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/separation/finetuned"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--valid-limit", type=int, default=100)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=200, help="lưu last.pt mỗi N bước cập nhật")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--keep-baseline-as-best", action="store_true",
                        help="chỉ lưu best.pt khi tốt hơn model gốc")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
