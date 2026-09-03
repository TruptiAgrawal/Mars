"""Generates and writes the ~15 hand-crafted mock CT cases (volume + metadata) used by the API and tests."""

import json
from pathlib import Path

from mars.data.synthetic import generate_volume
from mars.models import ClinicalMetadata

CASES_DIR = Path(__file__).parent / "cases"

SHAPE = (16, 32, 32)

# Each entry: (case_id, label, center, radius, intensity, background, noise_std, seed,
#              expected_site, expected_size_range_mm3, prior_history)
CASE_SPECS = [
    ("case_001", "Clean anterior lesion, high confidence", (4, 16, 16), 5.0, 200.0, 50.0, 2.0, 1, "anterior", (50.0, 5000.0), "none"),
    ("case_002", "Clean anterior lesion, matches history", (3, 12, 20), 4.5, 190.0, 50.0, 2.0, 2, "anterior", (30.0, 5000.0), "prior lung nodule"),
    ("case_003", "Clean posterior lesion, high confidence", (12, 16, 16), 5.0, 200.0, 50.0, 2.0, 3, "posterior", (50.0, 5000.0), "none"),
    ("case_004", "Clean posterior lesion, matches history", (13, 20, 10), 4.5, 195.0, 50.0, 2.0, 4, "posterior", (30.0, 5000.0), "none"),
    ("case_005", "Low-contrast anterior lesion", (4, 16, 16), 5.0, 128.0, 50.0, 2.0, 5, "anterior", (50.0, 5000.0), "none"),
    ("case_006", "Low-contrast posterior lesion", (12, 16, 16), 5.0, 126.0, 50.0, 2.0, 6, "posterior", (50.0, 5000.0), "none"),
    ("case_007", "Small low-contrast lesion", (4, 16, 16), 2.0, 130.0, 50.0, 2.0, 7, "anterior", (50.0, 5000.0), "none"),
    ("case_008", "Noisy borderline lesion", (4, 16, 16), 4.0, 150.0, 50.0, 20.0, 8, "anterior", (50.0, 5000.0), "none"),
    ("case_009", "Site mismatch: lesion posterior, expected anterior", (12, 16, 16), 5.0, 200.0, 50.0, 2.0, 9, "anterior", (50.0, 5000.0), "none"),
    ("case_010", "Site mismatch: lesion anterior, expected posterior", (4, 16, 16), 5.0, 200.0, 50.0, 2.0, 10, "posterior", (50.0, 5000.0), "none"),
    ("case_011", "Size mismatch: lesion too small for expected range", (4, 16, 16), 2.0, 200.0, 50.0, 2.0, 11, "anterior", (2000.0, 5000.0), "none"),
    ("case_012", "Size mismatch: lesion too large for expected range", (4, 16, 16), 7.0, 200.0, 50.0, 2.0, 12, "anterior", (1.0, 20.0), "none"),
    ("case_013", "Combined site and size mismatch", (12, 16, 16), 7.0, 200.0, 50.0, 2.0, 13, "anterior", (1.0, 20.0), "none"),
    ("case_014", "No detectable lesion (background only)", (4, 16, 16), 5.0, 55.0, 50.0, 2.0, 14, "anterior", (50.0, 5000.0), "none"),
    ("case_015", "Clean lesion, mid-size, borderline uncertainty", (4, 16, 16), 3.5, 145.0, 50.0, 5.0, 15, "anterior", (50.0, 5000.0), "none"),
]


def generate_all_fixtures() -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    for (case_id, label, center, radius, intensity, background, noise_std, seed,
         expected_site, expected_size_range_mm3, prior_history) in CASE_SPECS:
        volume = generate_volume(
            shape=SHAPE, center=center, radius=radius, intensity=intensity,
            background=background, noise_std=noise_std, seed=seed,
        )
        import numpy as np
        np.save(CASES_DIR / f"{case_id}.npy", volume)

        metadata = ClinicalMetadata(
            case_id=case_id, expected_site=expected_site,
            expected_size_range_mm3=expected_size_range_mm3, prior_history=prior_history,
        )
        meta_dict = metadata.model_dump()
        meta_dict["label"] = label
        with open(CASES_DIR / f"{case_id}.json", "w") as f:
            json.dump(meta_dict, f, indent=2)


if __name__ == "__main__":
    generate_all_fixtures()
    print(f"Generated {len(CASE_SPECS)} fixture cases in {CASES_DIR}")
