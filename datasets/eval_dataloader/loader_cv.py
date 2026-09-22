"""
BTCV-only, fold-aware evaluation loader for 5-fold cross-validation.

AbstractLoader.prepare_datalist() (loader_abstract.py) resolves its manifest
generically as:

    data_txt_path + dataset_list[i] + f"_{split}.txt"

With dataset_list=['01_BTCV'] and split='test', that resolves to
"01_BTCV_test.txt" -- which is the OLD, fixed 3-volume test split
(img0038-0040) left over from a different experiment, NOT this fold's 6
held-out volumes. Merely loading a fold's checkpoint with AbstractLoader
does not guarantee evaluation only touches that fold's held-out volumes.

FoldTestLoader fixes this by reading exactly 01_BTCV_test_4to1.txt from the
fold's data_txt_path (the same file cache_dataset3d_cv.py's leakage check
already reads on the training side) -- so evaluation is guaranteed to run on
precisely the 6 volumes that fold never trained on, regardless of whatever
dataset_list happens to be set in the config.
"""

import os

from .loader_abstract import AbstractLoader


class FoldTestLoader(AbstractLoader):

    def prepare_datalist(self):
        config = self.config
        test_file = os.path.join(config["data_txt_path"], "01_BTCV_test_4to1.txt")

        if not os.path.isfile(test_file):
            raise RuntimeError(f"Fold held-out manifest not found: {test_file}")

        data_paths = []
        with open(test_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                basename = os.path.basename(line.split()[0])
                volume_id = basename.replace(".nii.gz", "")   # img0001
                num = volume_id.replace("img", "")             # 0001
                img_path = config['data_root_path'] + f"01_BCV-Abdomen/Training/img/{volume_id}.nii.gz"
                label_path = config['data_root_path'] + f"01_BCV-Abdomen/Training/label/label{num}.nii.gz"
                data_paths.append({'img_path': img_path, 'label_path': label_path, 'name': volume_id})

        print(f"[FoldTestLoader] {test_file}: {len(data_paths)} held-out volumes")
        for d in data_paths:
            print("  ", d['name'])

        return data_paths
