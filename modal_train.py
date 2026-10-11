"""Optional Modal launcher for remote GPU training.

Modal supplies remote compute only. Jupiter's learning controller and model
configuration stay in the project files; no permanent model hyperparameters are
baked into this launcher.

Remote training is never part of the default run. It bills an A100 for up to an
hour, which is a decision for a person to make, not a side effect of pressing a
key. The entry point therefore calls this file with --check, which reports
whether remote training is available and exits without touching the cloud.

    python modal_train.py            run remote training
    python modal_train.py --check    report availability and exit
"""

import argparse
import os
import subprocess
import sys

REMOTE_DIR = "/root/Jupiter"
TRAIN_TIMEOUT = 3600
IMAGE_PACKAGES = ("torch", "numpy", "safetensors")
PYTHON_VERSION = "3.11"


def availability():
    """Whether remote training could run here, and why not when it cannot.

    modal is an optional dependency, so its absence is a normal state to report
    rather than a crash. Nothing is imported at module level, which is what
    allows --check to work on a machine that has never installed it.
    """
    try:
        import modal
    except ImportError:
        return (
            False,
            "modal is not installed; remote training is skipped. It is "
            "optional: pip install modal, then python modal_train.py",
        )
    return True, f"modal {getattr(modal, '__version__', 'unknown')} is installed"


def build_app():
    """Build the Modal app. Only called when remote training is actually wanted."""
    import modal

    app = modal.App("jupiter-learning")
    image = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*IMAGE_PACKAGES)
        .add_local_dir(".", remote_path=REMOTE_DIR)
    )

    @app.function(image=image, gpu="any", timeout=TRAIN_TIMEOUT)
    def learn_remote():
        os.chdir(REMOTE_DIR)
        subprocess.run(
            [sys.executable, "learning_worker.py", "--once"], check=True
        )

    @app.local_entrypoint()
    def entry():
        learn_remote.remote()

    return app, entry


def main():
    parser = argparse.ArgumentParser(
        description="Run Jupiter's training on a remote Modal GPU.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether remote training is available and exit",
    )
    args = parser.parse_args()
    available, reason = availability()
    print(f"[remote] {reason}")
    if args.check or not available:
        return
    _, entry = build_app()
    entry()


if __name__ == "__main__":
    main()
