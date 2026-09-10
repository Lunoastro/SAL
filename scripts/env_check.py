import argparse
import importlib
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("PROJECT_ROOT", str(REPO))

PACKAGES = (
    "torch",
    "torchaudio",
    "lightning",
    "pytorch-lightning",
    "s3prl",
    "hydra-core",
    "omegaconf",
    "torchmetrics",
    "numpy",
)
MODULES = (
    "torch",
    "torchaudio",
    "lightning",
    "s3prl",
    "src.train",
    "src.eval",
    "src.trainers.SAL",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAL Environment and Checker"
    )
    parser.add_argument(
        "--mode",
        choices=["import", "gpu", "preflight"],
        default="import",
        help=(
            "'import' checks the Python environment; 'gpu' also requires a "
            "working CUDA device; 'preflight' adds training path checks"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for each isolated module import subprocess check (default: 30s)",
    )
    return parser.parse_args()


def check_preflight(failures):
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=REPO / ".env", override=False)
    except Exception as exc:
        failures.append(f"Failed to load .env: {exc}")
        return

    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        failures.append("PROJECT_ROOT environment variable is not set")
    elif not Path(project_root).is_dir():
        failures.append(f"PROJECT_ROOT directory does not exist: {project_root}")
    else:
        print(f"OK: PROJECT_ROOT found at {project_root}")

    xlsr_path = os.environ.get("XLSR_CHECKPOINT")
    if not xlsr_path:
        failures.append("XLSR_CHECKPOINT environment variable is not set")
    else:
        path = Path(xlsr_path)
        if not path.is_file():
            failures.append(f"XLSR_CHECKPOINT file does not exist: {xlsr_path}")
        elif path.stat().st_size == 0:
            failures.append(f"XLSR_CHECKPOINT file is empty (0 bytes): {xlsr_path}")
        else:
            print(f"OK: XLSR_CHECKPOINT ({path.stat().st_size / (1024 * 1024):.1f} MB)")

    ps_root = os.environ.get("PARTIALSPOOF_ROOT")
    if not ps_root:
        failures.append("PARTIALSPOOF_ROOT environment variable is not set")
    else:
        path = Path(ps_root)
        if not path.is_dir():
            failures.append(f"PARTIALSPOOF_ROOT directory does not exist: {ps_root}")
        else:
            required_dirs = ("train", "dev", "eval", "segment_labels")
            missing_dirs = [name for name in required_dirs if not (path / name).is_dir()]
            if missing_dirs:
                failures.append(
                    "PARTIALSPOOF_ROOT is missing directories: "
                    + ", ".join(missing_dirs)
                )
            else:
                print(f"OK: PARTIALSPOOF_ROOT found at {ps_root}")

    log_dir = os.environ.get("SAL_LOG_DIR")
    if not log_dir:
        failures.append("SAL_LOG_DIR environment variable is not set")
    else:
        path = Path(log_dir)
        existing_path = path
        while not existing_path.exists() and existing_path != existing_path.parent:
            existing_path = existing_path.parent
        if not existing_path.is_dir() or not os.access(existing_path, os.W_OK | os.X_OK):
            failures.append(
                f"SAL_LOG_DIR cannot be created or written: {log_dir}"
            )
        else:
            print(f"OK: SAL_LOG_DIR has a writable parent at {existing_path}")

    lps_root = os.environ.get("LLAMASPOOF_ROOT")
    if lps_root:
        path = Path(lps_root)
        if not path.is_dir() or not (path / "segment_labels").is_dir():
            failures.append(f"LLAMASPOOF_ROOT is incomplete: {lps_root}")
        else:
            print(f"OK: LLAMASPOOF_ROOT found at {lps_root}")

    eval_checkpoint = os.environ.get("SAL_EVAL_CHECKPOINT")
    if eval_checkpoint and not Path(eval_checkpoint).is_file():
        failures.append(
            f"SAL_EVAL_CHECKPOINT file does not exist: {eval_checkpoint}"
        )


def main():
    args = parse_args()
    failures = []

    print(f"System: {platform.platform()}, {platform.processor()}, {platform.python_implementation()}")
    if sys.version_info[:2] != (3, 10):
        failures.append(
            f"Expected Python 3.10, found {sys.version_info[0]}.{sys.version_info[1]}"
        )

    print("\n--- Checking Package Metadata ---")
    for package in PACKAGES:
        try:
            print(f"  {package}: {version(package)}")
        except PackageNotFoundError:
            failures.append(f"Package missing: {package}")

    print("\n--- Checking Isolated Module Imports ---")
    for module in MODULES:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import importlib; importlib.import_module({module!r})",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            if result.returncode == 0:
                print(f"  IMPORT OK: {module}")
            else:
                details = result.stderr.strip() or result.stdout.strip()
                failures.append(f"Import failed: {module}\n{details}")
        except subprocess.TimeoutExpired:
            failures.append(f"Import timed out after {args.timeout}s: {module}")

    print("\n--- Checking CUDA and Hardware Accelerator ---")
    try:
        torch = importlib.import_module("torch")
    except Exception:
        torch = None

    require_gpu = args.mode in ("gpu", "preflight")

    if torch is not None:
        cuda_build = getattr(torch.version, "cuda", None)
        print(f"  PyTorch CUDA build: {cuda_build}")
        try:
            available = torch.cuda.is_available()
            print(f"  CUDA available: {available}")
            if available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory
                probe = torch.ones(1, device="cuda") + 1
                torch.cuda.synchronize()
                print(
                    f"  GPU Count: {device_count}, Device 0: {device_name}, "
                    f"memory: {total_memory / (1024 ** 3):.1f} GiB, "
                    f"CUDA probe: {probe.item():.0f}"
                )
            elif require_gpu:
                failures.append("CUDA is not available, but GPU execution was requested or required.")
        except Exception as exc:
            failures.append(f"CUDA inspection failed: {exc}")
    else:
        failures.append("torch module could not be imported")

    if args.mode == "preflight":
        print("\n--- Running Training Path & File Verification ---")
        check_preflight(failures)

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
