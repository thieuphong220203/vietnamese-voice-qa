import platform
from pathlib import Path


DRIVE_ROOT = Path("/content/drive/MyDrive/AdvancedML")
DIRECTORIES = (
    DRIVE_ROOT / "data" / "raw",
    DRIVE_ROOT / "data" / "processed",
    DRIVE_ROOT / "data" / "samples",
    DRIVE_ROOT / "checkpoints",
    DRIVE_ROOT / "outputs",
)


def setup_google_drive() -> None:
    try:
        from google.colab import drive
    except ImportError:
        print("Google Drive setup skipped: this script is not running in Colab.")
        return

    drive.mount("/content/drive")

    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ready: {directory}")


def main() -> None:
    print("Hello from Vietnamese Voice QA!")
    print(f"Python: {platform.python_version()}")
    setup_google_drive()
    print("Local -> GitHub -> Google Colab flow is working.")


if __name__ == "__main__":
    main()
