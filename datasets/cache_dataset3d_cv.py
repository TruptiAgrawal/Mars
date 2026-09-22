"""
BTCV-only, fold-aware cached dataset for 5-fold cross-validation.

Unlike cache_dataset3d_e2.py (the BTCV + WORD-50 E2 experiment), this loader:
    - Reads ONLY 01_BTCV_train_4to1.txt from the fold's data_txt_path.
    - Never reads, requires, or references WORD_50_train.txt.
    - Scans the SHARED btcv_cross_validation/cache/all30/ cache (built once,
      from all 30 BTCV volumes) and selects only the cached samples whose
      source volume belongs to the current fold's 24 training volumes.
    - Also reads that fold's 01_BTCV_test_4to1.txt (held-out volumes) purely
      as a safety check, and hard-fails if any selected training sample
      belongs to a held-out volume -- making leakage detectable rather than
      merely assumed away.

This subclasses cache_dataset3d3.Dataset3D (not cache_dataset3d.Dataset3D):
cache_dataset3d3.py's _get_cached_data reads the image/*.jpg + label_jpg/*.jpg
+ meta/*.npy triplet that the caching step (cache_dataset3d.py) actually
produced on disk. cache_dataset3d.py's own _get_cached_data expects a
different, legacy .npz-based layout that was never written for this cache.
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
        # Read the current fold's train/held-out BTCV volume IDs.
        # ------------------------------------------------------------
        train_file = os.path.join(config["data_txt_path"], "01_BTCV_train_4to1.txt")
        test_file = os.path.join(config["data_txt_path"], "01_BTCV_test_4to1.txt")

        def read_ids(path):
            ids = set()
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    image_path = line.split()[0]
                    basename = os.path.basename(image_path)
                    # img0001.nii.gz -> img0001
                    ids.add(basename.replace(".nii.gz", ""))
            return ids

        train_ids = read_ids(train_file)

        if not os.path.isfile(test_file):
            raise RuntimeError(
                f"Held-out manifest not found (required for the leakage check): {test_file}"
            )
        held_out_ids = read_ids(test_file)

        manifest_overlap = train_ids & held_out_ids
        if manifest_overlap:
            raise RuntimeError(
                f"CV fold leakage in the manifests themselves (train/test overlap): "
                f"{sorted(manifest_overlap)}"
            )

        # ------------------------------------------------------------
        # Scan the shared all30 cache and select only this fold's
        # training-volume samples.
        # ------------------------------------------------------------
        selected_paths = []
        sample_counts = {}
        leaked_samples = []

        meta_paths = sorted(
            glob.glob(os.path.join(cache_root, "01_BCV-Abdomen", "meta", "*.npy"))
        )

        for meta_path in meta_paths:
            meta = np.load(meta_path, allow_pickle=True).tolist()

            img_name = meta["img_name"]
            basename = os.path.basename(img_name)
            volume_id = basename.replace(".nii.gz", "")

            if volume_id not in train_ids:
                continue

            # meta_XXXX_YYYYYYYY.npy
            filename = os.path.basename(meta_path)
            parts = filename.replace(".npy", "").split("_")
            if len(parts) != 3:
                raise RuntimeError(f"Unexpected BTCV metadata filename: {meta_path}")

            cache_index = parts[1]
            sample_id = parts[2]

            mask_pattern = os.path.join(
                cache_root, "01_BCV-Abdomen", "label_jpg",
                f"label_{cache_index}_{sample_id}_*.jpg",
            )
            matches = sorted(glob.glob(mask_pattern))

            for mask_path in matches:
                selected_paths.append(mask_path)
                if volume_id in held_out_ids:
                    leaked_samples.append(mask_path)

            sample_counts[volume_id] = sample_counts.get(volume_id, 0) + len(matches)

        selected_paths = sorted(selected_paths)

        # ------------------------------------------------------------
        # Integrity checks.
        # ------------------------------------------------------------
        missing_train = train_ids - set(sample_counts)
        if missing_train:
            raise RuntimeError(
                f"BTCV training volumes missing cached samples: {sorted(missing_train)}"
            )

        if leaked_samples:
            raise RuntimeError(
                f"LEAKAGE DETECTED: {len(leaked_samples)} selected training sample(s) "
                f"belong to held-out volumes. Examples: {sorted(set(leaked_samples))[:5]}"
            )

        # ------------------------------------------------------------
        # Dataset audit.
        # ------------------------------------------------------------
        def sort_key(vid):
            return int("".join(c for c in vid if c.isdigit()))

        print()
        print("=" * 70)
        print("CV DATASET AUDIT")
        print("=" * 70)
        print(f"Fold data_txt_path      : {config['data_txt_path']}")
        print(f"Training volumes        : {len(train_ids)}")
        print(f"Held-out volumes        : {len(held_out_ids)}")
        print()
        print(f"Cached training samples : {len(selected_paths)}")
        print(f"Leaked samples found    : {len(leaked_samples)} (must be 0)")
        print()
        print("Training volume IDs:")
        print(sorted(train_ids, key=sort_key))
        print()
        print("Held-out volume IDs:")
        print(sorted(held_out_ids, key=sort_key))
        print()
        print("First 3 selected cache samples:")
        for path in selected_paths[:3]:
            print(" ", path)
        print("Last 3 selected cache samples:")
        for path in selected_paths[-3:]:
            print(" ", path)
        print("=" * 70)
        print("CV DATASET AUDIT PASSED -- zero held-out samples selected")
        print("=" * 70)
        print()

        return selected_paths
