import json
import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from dataset.dataset_mixin import DataAugmentationMixin, LoadScannetppMixin
from dataset.path_config import SCAN_FAMILY_BASE
from torch.utils.data import Dataset
from utils.label_utils import LabelConverter
from collections import Counter


class OVFGVGDataset(Dataset, LoadScannetppMixin, DataAugmentationMixin):
    def __init__(self, scene_dir, prompt_file, max_obj_len=60, num_points=1024):
        # load file
        self.data = self._import_data(prompt_file)
        anno_file = prompt_file
        self.scan_ids = set(self.data.scene_id.tolist())  # scan ids in data

        # fill parameters
        self.max_obj_len = max_obj_len - 1
        self.num_points = num_points

        # lod category file
        self.int2cat = json.load(
            open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), "r")
        )
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(
            os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2-labels.combined.tsv")
        )

        # load scans
        self.scans = self.load_scannetpp(scene_dir, self.scan_ids)

        # build unique multiple look up
        for scan_id in self.scan_ids:
            cache = {}
            label_list = []
            for item in self.data.itertuples(index=False, name="Prompt"):
                if item.scene_id == scan_id and item.target_id not in cache.keys():
                    cache[item.target_id] = 1
                    # label_list.append(self.label_converter.id_to_scannetid[self.cat2int[item.target_label]])
                    label_list.append(item.target_label)
            self.scans[scan_id]["label_count"] = Counter(label_list)

    def __len__(self):
        return len(self.data)

    def _import_data(self, data_path: str):
        # load prompts
        prompts = pd.read_csv(data_path, dtype={"target_id": str})

        return prompts

    def __getitem__(self, idx):
        # load scanrefer
        item = self.data.iloc[idx]
        item_id = item["prompt_id"]
        scan_id = item["scene_id"]
        sentence = item["prompt"]

        # load pcds and labels
        obj_pcds = deepcopy(self.scans[scan_id]["pcds"])  # N, 6
        obj_labels = deepcopy(self.scans[scan_id]["inst_labels"])  # N

        # filter out background or language
        # do not filter for predicted labels, because these labels are not accurate
        selected_obj_idxs = [i for i in range(len(obj_pcds))]
        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]
        assert len(obj_pcds) == len(obj_labels)

        # crop objects
        if self.max_obj_len < len(obj_labels):
            logging.warning(
                f"Expected no more than {self.max_obj_len} objects, but trying to process a scene with "
                f"{len(obj_labels)} objects."
            )

        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            # build locs
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            # build box
            obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
            obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))
            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3] ** 2, 1)))
            if max_dist < 1e-6:  # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))

        assert obj_fts.shape[0] == obj_locs.shape[0]

        data_dict = {
            "sentence": sentence,
            "obj_fts": obj_fts,  # N, 6
            "obj_locs": obj_locs,  # N, 3
            "obj_boxes": obj_boxes,  # N, 6
            "data_idx": item_id,
            "scene_id": scan_id,
        }

        return data_dict
