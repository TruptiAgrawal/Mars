"""
E2 dataset for BTCV + WORD-50.

Scientific purpose:
    Isolated cached dataset implementation for the E2 experiment.

Selection:
    - BTCV: exactly the 24 frozen E1 training volumes
    - WORD: exactly the 50 volumes listed in WORD_50_train.txt

This file intentionally does NOT modify cache_dataset3d3.py.
"""

import glob
import os
import numpy as np

from .cache_dataset3d3 import Dataset3D as BaseDataset3D


class Dataset3D(BaseDataset3D):

    def prepare_cached_datalist(self):
        config = self.config
        cache_root = config["cache_data_path"]

        # ------------------------------------------------------------
        # Read the exact BTCV E1 training volume IDs.
        # ------------------------------------------------------------
        btcv_train_file = os.path.join(
            config["data_txt_path"],
            "01_BTCV_train_4to1.txt",
        )

        btcv_ids = set()

        with open(btcv_train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                image_path = line.split()[0]
                basename = os.path.basename(image_path)

                # img0001.nii.gz -> img0001
                volume_id = basename.replace(".nii.gz", "")
                btcv_ids.add(volume_id)

        # ------------------------------------------------------------
        # Read the exact WORD-50 volume IDs.
        # ------------------------------------------------------------
        word_train_file = os.path.join(
            config["data_txt_path"],
            "WORD_50_train.txt",
        )

        word_ids = set()

        with open(word_train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                image_path = line.split()[0]
                basename = os.path.basename(image_path)

                # word_0002.nii.gz -> 0002
                volume_id = basename.replace("word_", "").replace(".nii.gz", "")
                word_ids.add(volume_id)

        # ------------------------------------------------------------
        # Scan cache metadata and select samples by original volume.
        # ------------------------------------------------------------
        selected_paths = []

        btcv_sample_counts = {}
        word_sample_counts = {}

        # BTCV
        btcv_meta_paths = sorted(
            glob.glob(os.path.join(cache_root, "01_BCV-Abdomen", "meta", "*.npy"))
        )

        for meta_path in btcv_meta_paths:
            meta = np.load(meta_path, allow_pickle=True).tolist()

            img_name = meta["img_name"]
            basename = os.path.basename(img_name)
            volume_id = basename.replace(".nii.gz", "")

            if volume_id not in btcv_ids:
                continue

            # meta_XXXX_YYYYYYYY.npy
            filename = os.path.basename(meta_path)
            parts = filename.replace(".npy", "").split("_")

            if len(parts) != 3:
                raise RuntimeError(
                    f"Unexpected BTCV metadata filename: {meta_path}"
                )

            cache_index = parts[1]
            sample_id = parts[2]

            mask_dir = os.path.join(
                cache_root,
                "01_BCV-Abdomen",
                "label_jpg",
            )

            mask_pattern = os.path.join(
                mask_dir,
                f"label_{cache_index}_{sample_id}_*.jpg",
            )

            matches = sorted(glob.glob(mask_pattern))

            for mask_path in matches:
                selected_paths.append(mask_path)

            btcv_sample_counts[volume_id] = (
                btcv_sample_counts.get(volume_id, 0) + len(matches)
            )

        # WORD
        word_meta_paths = sorted(
            glob.glob(os.path.join(cache_root, "07_WORD", "meta", "*.npy"))
        )

        for meta_path in word_meta_paths:
            meta = np.load(meta_path, allow_pickle=True).tolist()

            img_name = meta["img_name"]
            basename = os.path.basename(img_name)

            # word_0002.nii.gz -> 0002
            volume_id = basename.replace("word_", "").replace(".nii.gz", "")

            if volume_id not in word_ids:
                continue

            filename = os.path.basename(meta_path)
            parts = filename.replace(".npy", "").split("_")

            if len(parts) != 3:
                raise RuntimeError(
                    f"Unexpected WORD metadata filename: {meta_path}"
                )

            cache_index = parts[1]
            sample_id = parts[2]

            mask_dir = os.path.join(
                cache_root,
                "07_WORD",
                "label_jpg",
            )

            mask_pattern = os.path.join(
                mask_dir,
                f"label_{cache_index}_{sample_id}_*.jpg",
            )

            matches = sorted(glob.glob(mask_pattern))

            for mask_path in matches:
                selected_paths.append(mask_path)

            word_sample_counts[volume_id] = (
                word_sample_counts.get(volume_id, 0) + len(matches)
            )

        # ------------------------------------------------------------
        # Deterministic ordering.
        # ------------------------------------------------------------
        selected_paths = sorted(selected_paths)

        # ------------------------------------------------------------
        # Integrity checks.
        # ------------------------------------------------------------
        found_btcv = set(btcv_sample_counts)
        found_word = set(word_sample_counts)

        missing_btcv = btcv_ids - found_btcv
        missing_word = word_ids - found_word

        if missing_btcv:
            raise RuntimeError(
                f"BTCV training volumes missing cached samples: "
                f"{sorted(missing_btcv)}"
            )

        if missing_word:
            raise RuntimeError(
                f"WORD-50 volumes missing cached samples: "
                f"{sorted(missing_word)}"
            )

        # ------------------------------------------------------------
        # Dataset audit.
        # ------------------------------------------------------------
        print()
        print("=" * 70)
        print("E2 DATASET AUDIT")
        print("=" * 70)

        print(f"BTCV training volumes : {len(btcv_ids)}")
        print(f"WORD training volumes : {len(word_ids)}")

        print()
        print(f"BTCV cached samples   : {sum(btcv_sample_counts.values())}")
        print(f"WORD cached samples   : {sum(word_sample_counts.values())}")
        print(f"TOTAL cached samples  : {len(selected_paths)}")

        print()
        print(
            "WORD sample-count range: "
            f"{min(word_sample_counts.values())} - "
            f"{max(word_sample_counts.values())}"
        )

        print()
        print("BTCV volume IDs:")
        print(sorted(btcv_ids, key=lambda x: int(
            "".join(c for c in x if c.isdigit())
        )))

        print()
        print("WORD-50 volume IDs:")
        print(sorted(word_ids, key=int))

        print()
        print("First 3 selected cache samples:")
        for path in selected_paths[:3]:
            print(" ", path)

        print()
        print("Last 3 selected cache samples:")
        for path in selected_paths[-3:]:
            print(" ", path)

        print("=" * 70)
        print("E2 DATASET AUDIT PASSED")
        print("=" * 70)
        print()

        return selected_paths
