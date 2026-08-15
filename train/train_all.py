"""Convenience script: train ALL cancer models in sequence."""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "train_skin_melanoma.py",
    "train_breast_cancer.py",
    "train_esophageal.py",
    "train_lung_cancer.py",
    "train_ovarian.py",
    "train_cervical.py",
    "train_pancreatic.py",
    "train_leukemia.py",
    "train_colorectal.py"
]

def main():
    for s in SCRIPTS:
        path = os.path.join(HERE, s)
        print("\n" + "#" * 80)
        print(f"# RUNNING: {s}")
        print("#" * 80)
        result = subprocess.run([sys.executable, path], cwd=HERE)
        if result.returncode != 0:
            print(f"❌ {s} failed with exit code {result.returncode}")
        else:
            print(f"✅ {s} completed successfully")

if __name__ == "__main__":
    main()