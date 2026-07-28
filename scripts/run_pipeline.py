from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    input_path = ROOT / "data" / "Consultant_Loyalty_Dataset_200_Customers.xlsx"
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    commands = [
        [sys.executable, "-m", "loyalty_engine.cli", "ingest", "--input", str(input_path)],
        [sys.executable, "-m", "loyalty_engine.cli", "features", "--input", str(input_path)],
        [sys.executable, "-m", "loyalty_engine.cli", "segment", "--input", str(input_path)],
        [sys.executable, "-m", "loyalty_engine.cli", "train", "--input", str(input_path)],
        [sys.executable, "-m", "loyalty_engine.cli", "score", "--input", str(input_path)],
    ]

    for command in commands:
        print("\n>>>", " ".join(command))
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
