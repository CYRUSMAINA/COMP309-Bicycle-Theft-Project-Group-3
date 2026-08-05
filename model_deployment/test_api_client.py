"""Send untouched test records to a running Flask prediction API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES = BASE_DIR / "artifacts" / "test_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--samples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = json.loads(DEFAULT_SAMPLES.read_text(encoding="utf-8"))

    health_response = requests.get(f"{args.base_url}/health", timeout=10)
    health_response.raise_for_status()
    print("Health:", health_response.json())

    for number, sample in enumerate(samples[: args.samples], start=1):
        response = requests.post(
            f"{args.base_url}/predict",
            json=sample["features"],
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        print(
            f"Sample {number}: expected={sample['expected_class']}, "
            f"predicted={result['prediction']}, "
            f"recovery_probability={result['recovery_probability']:.4f}"
        )


if __name__ == "__main__":
    main()
