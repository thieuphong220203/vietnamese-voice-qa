from __future__ import annotations

from scripts.evaluate_integrated_ablation import make_detail, summarize, summarize_by_condition


def test_summary_uses_corpus_error_rates_and_duration_weighted_rtf() -> None:
    metadata = {
        "id": "sample_1",
        "transcript": "xin chào bạn",
        "snr_db": "0",
        "sir_db": "2",
        "noise_type": "DEMAND_PCAFE",
    }
    first = make_detail(
        metadata=metadata,
        config="C1",
        hypothesis="xin chào",
        duration=2.0,
        separation_s=1.0,
        enhancement_s=0.0,
        asr_s=0.5,
        selected_index=0,
        oracle_index=0,
    )
    second_metadata = {**metadata, "id": "sample_2", "transcript": "một hai"}
    second = make_detail(
        metadata=second_metadata,
        config="C1",
        hypothesis="một ba",
        duration=3.0,
        separation_s=2.0,
        enhancement_s=0.0,
        asr_s=1.0,
        selected_index=1,
        oracle_index=0,
    )

    (summary,) = summarize([first, second], ["C1"])

    # 2 lỗi / 5 từ tham chiếu; RTF = (1 + 2) / (2 + 3).
    assert float(summary["phowhisper_small_wer_percent"]) == 40.0
    assert float(summary["preprocessing_rtf"]) == 0.6
    assert float(summary["mean_latency_s"]) == 2.25
    assert float(summary["target_selection_accuracy_percent"]) == 50.0

    conditions = summarize_by_condition([first, second], ["C1"])
    assert len(conditions) == 1
    assert conditions[0]["snr_db"] == "0"
    assert conditions[0]["sir_db"] == "2"
    assert conditions[0]["samples"] == 2


def test_c0_has_zero_preprocessing_and_no_selection_accuracy() -> None:
    detail = make_detail(
        metadata={
            "id": "sample",
            "transcript": "câu đúng",
            "snr_db": "5",
            "sir_db": "3",
            "noise_type": "DEMAND_TCAR",
        },
        config="C0",
        hypothesis="câu đúng",
        duration=4.0,
        separation_s=0.0,
        enhancement_s=0.0,
        asr_s=0.25,
        selected_index=None,
        oracle_index=None,
    )

    (summary,) = summarize([detail], ["C0"])
    assert float(summary["phowhisper_small_wer_percent"]) == 0.0
    assert float(summary["preprocessing_rtf"]) == 0.0
    assert float(summary["mean_latency_s"]) == 0.25
    assert summary["target_selection_accuracy_percent"] == ""
