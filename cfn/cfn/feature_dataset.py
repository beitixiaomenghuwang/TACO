
import numpy as np
import torch
import os
from tqdm import tqdm

class CoinFlipMaker:
    def __init__(self, output_dimensions=20, only_zero_flips=False):
        self.output_dimensions = output_dimensions
        self.only_zero_flips = only_zero_flips

    def __call__(self, seed):
        if self.only_zero_flips:
            return np.zeros((self.output_dimensions), dtype=np.float32)
        rng = np.random.RandomState(seed)
        return 2 * rng.binomial(1, 0.5, size=(self.output_dimensions)) - 1


def load_all_pt_files(folder_path, recursive=False, map_location="cpu"):
    pt_files = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pt"):
                    pt_files.append(os.path.join(root, file))
    else:
        pt_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".pt")
        ]

    results = {}
    for file_path in tqdm(pt_files, desc="📦 Loading .pt files"):
        try:
            data = torch.load(file_path, map_location=map_location)
            rel_name = os.path.relpath(file_path, folder_path)
            results[rel_name] = data
        except Exception as e:
            print(f"\n❌ Failed to load {file_path}: {e}")

    print(f"✅ load {len(results)}  .pt files")
    return results


class cfn_feature_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        feature_dir,
        multi_feature_file = False,
    ):
        if multi_feature_file:
            feature_dict = load_all_pt_files(feature_dir)
            features = []

            for name, feats in feature_dict.items():
                features.extend(feats)  # if list，extend
            
        else:
            features = torch.load(feature_dir, map_location="cpu")
        self.CoinFlipMaker = CoinFlipMaker()
        self.features = features
        self.len = len(features)

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> dict:
        item = {}
        item["feature"] = self.features[idx]
        item["CoinFlip_target"] = self.CoinFlipMaker(idx)

        return item

