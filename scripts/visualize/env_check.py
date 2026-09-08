import importlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PACKAGES = ("torch","torchaudio","lightning","pytorch-lightning","s3prl","hydra-core","omegaconf","torchmetrics","numpy",)
MODULES = (
    "torch",
    "torchaudio",
    "lightning",
    "s3prl",
    "src.train",
    "src.eval",
    "src.trainers.SAL",
)

def main(): 
    failures = [] 
    print(f"{platform.platform()}, {platform.processor()}, {platform.python_implementation()}")
    if sys.version_info[:2] != (3, 10):
        failures.append("Expected Python 3.10")

    for package in PACKAGES:
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            failures.append(f"Package missing: {package}")

    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"IMPORT OK: {module}")
        except Exception as exc:
            failures.append(
                f"Import failed: {module}: {type(exc).__name__}: {exc}"
            )

    torch = sys.modules.get("torch")
    if torch is not None:
        print(f"PyTorch CUDA build: {torch.version.cuda}")
        try:
            available = torch.cuda.is_available()
            print(f"CUDA available: {available}")
            if available:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            failures.append(f"CUDA inspection failed: {exc}")

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
if __name__ == "__main__":
    raise SystemExit(main())