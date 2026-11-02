"""Create the deterministic synthetic H&E-like image used in the walkthrough."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from validate_dip_algorithms import synthetic_he_image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "demo_data" / "synthetic_he_demo.png"


def main() -> None:
    rgb, _, _, _ = synthetic_he_image(height=384, width=512, seed=20261102)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(OUTPUT, format="PNG", optimize=True)
    print(f"Wrote deterministic synthetic image: {OUTPUT.relative_to(ROOT)} ({rgb.shape[1]}x{rgb.shape[0]}, seed=20261102)")


if __name__ == "__main__":
    main()
