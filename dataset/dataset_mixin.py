from dataset.path_config import SCAN_FAMILY_BASE, MASK_BASE
import json
import logging
import os
import numpy as np
import torch
from scipy import sparse
import plyfile

from utils.eval_helper import convert_pc_to_box

# rotate angles
ROTATE_ANGLES = [0, np.pi / 2, np.pi, np.pi * 3 / 2]


class LoadScannetMixin(object):
    def __init__(self):
        pass

    def load_scannet(self, scan_ids, pc_type, load_inst_info):
        scans = {}
        # attribute
        # inst_labels, inst_locs, inst_colors, pcds, / pcds_pred, inst_labels_pred
        for scan_id in scan_ids:
            # load inst
            if load_inst_info:
                inst_labels = json.load(
                    open(os.path.join(SCAN_FAMILY_BASE, "scan_data", "instance_id_to_name", "%s.json" % scan_id))
                )
                inst_labels = [self.cat2int[i] for i in inst_labels]
                inst_locs = np.load(
                    os.path.join(SCAN_FAMILY_BASE, "scan_data", "instance_id_to_loc", "%s.npy" % scan_id)
                )
                inst_colors = json.load(
                    open(os.path.join(SCAN_FAMILY_BASE, "scan_data", "instance_id_to_gmm_color", "%s.json" % scan_id))
                )
                inst_colors = [
                    np.concatenate([np.array(x["weights"])[:, None], np.array(x["means"])], axis=1).astype(np.float32)
                    for x in inst_colors
                ]
                scans[scan_id] = {
                    "inst_labels": inst_labels,  # (n_obj, )
                    "inst_locs": inst_locs,  # (n_obj, 6) center xyz, whl
                    "inst_colors": inst_colors,  # (n_obj, 3x4) cluster * (weight, mean rgb)
                }
            else:
                scans[scan_id] = {}

            # load pcd data
            pcd_data = torch.load(
                os.path.join(SCAN_FAMILY_BASE, "scan_data", "pcd_with_global_alignment", "%s.pth" % scan_id)
            )
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            # convert to gt object
            if load_inst_info:
                obj_pcds = []
                for i in range(instance_labels.max() + 1):
                    mask = instance_labels == i  # time consuming
                    obj_pcds.append(pcds[mask])
                scans[scan_id]["pcds"] = obj_pcds
                # calculate box for matching
                obj_center = []
                obj_box_size = []
                for i in range(len(obj_pcds)):
                    c, b = convert_pc_to_box(obj_pcds[i])
                    obj_center.append(c)
                    obj_box_size.append(b)
                scans[scan_id]["obj_center"] = obj_center
                scans[scan_id]["obj_box_size"] = obj_box_size

            # load mask
            if pc_type == "pred":
                """
                obj_mask_path = os.path.join(os.path.join(SCAN_FAMILY_BASE, 'mask'), str(scan_id) + ".mask" + ".npy")
                obj_label_path = os.path.join(os.path.join(SCAN_FAMILY_BASE, 'mask'), str(scan_id) + ".label" + ".npy")
                obj_pcds = []
                obj_mask = np.load(obj_mask_path)
                obj_labels = np.load(obj_label_path)
                obj_labels = [self.label_converter.nyu40id_to_id[int(l)] for l in obj_labels]
                """
                obj_mask_path = os.path.join(MASK_BASE, str(scan_id) + ".mask" + ".npz")
                obj_label_path = os.path.join(MASK_BASE, str(scan_id) + ".label" + ".npy")
                obj_pcds = []
                obj_mask = np.array(sparse.load_npz(obj_mask_path).todense())[:50, :]
                obj_labels = np.load(obj_label_path)[:50]
                for i in range(obj_mask.shape[0]):
                    mask = obj_mask[i]
                    if pcds[mask == 1, :].shape[0] > 0:
                        obj_pcds.append(pcds[mask == 1, :])
                scans[scan_id]["pcds_pred"] = obj_pcds
                scans[scan_id]["inst_labels_pred"] = obj_labels[: len(obj_pcds)]
                # calculate box for pred
                obj_center_pred = []
                obj_box_size_pred = []
                for i in range(len(obj_pcds)):
                    c, b = convert_pc_to_box(obj_pcds[i])
                    obj_center_pred.append(c)
                    obj_box_size_pred.append(b)
                scans[scan_id]["obj_center_pred"] = obj_center_pred
                scans[scan_id]["obj_box_size_pred"] = obj_box_size_pred
        print("finish loading scannet data")
        return scans


class DataAugmentationMixin(object):
    def __init__(self):
        pass

    def build_rotate_mat(self):
        theta_idx = np.random.randint(len(ROTATE_ANGLES))
        theta = ROTATE_ANGLES[theta_idx]
        if (theta is not None) and (theta != 0) and (self.split == "train"):
            rot_matrix = np.array(
                [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float32
            )
        else:
            rot_matrix = None
        return rot_matrix

    def random_flip(self, point_cloud, p):
        ret_point_cloud = point_cloud.copy()
        if np.random.rand() < p:
            ret_point_cloud[:, 0] = point_cloud[:, 0]
        if np.random.rand() < p:
            ret_point_cloud[:, 1] = point_cloud[:, 1]
        return ret_point_cloud

    def set_color_to_zero(self, point_cloud, p):
        ret_point_cloud = point_cloud.copy()
        if np.random.rand() < p:
            ret_point_cloud[:, 3:6] = 0
        return ret_point_cloud

    def random_jitter(self, point_cloud):
        noise_sigma = 0.01
        noise_clip = 0.05
        jitter = np.clip(noise_sigma * np.random.randn(point_cloud.shape[0], 3), -noise_clip, noise_clip)
        ret_point_cloud = point_cloud.copy()
        ret_point_cloud[:, 0:3] += jitter
        return ret_point_cloud


class LoadScannetppMixin:
    def __init__(self):
        pass

    def _load_pcd(self, scene_file):
        try:
            raw_points = plyfile.PlyData().read(scene_file)
        except plyfile.PlyHeaderParseError:
            logging.error("Could not parse mesh for scene due to corrupt header")
            return None
        except plyfile.PlyElementParseError:
            logging.error("Could not parse mesh for scene due to element parsing error")
            return None
        except FileNotFoundError:
            logging.error(f"Could not find .ply file: {scene_file}")
            return None

        vertices = np.array([list(x) for x in raw_points.elements[0]])  # num_points x 6
        points, colors = vertices[:, :3], vertices[:, 3:6]
        colors = colors / 127.5 - 1
        return np.concatenate([points, colors], 1)

    def _extract_instance_annotations(self, vertices, segments_data, annotations):
        segments = np.array(segments_data["segIndices"])

        pcds = []
        centers = []
        box_sizes = []
        labels = []
        for obj in annotations["segGroups"]:
            label = "_".join(obj["label"].split())
            segments_obj = obj["segments"]

            # vertices
            v_mask = np.isin(segments, segments_obj)
            pcd = vertices[v_mask]

            labels.append(label)
            pcds.append(pcd)
            centers.append(obj["obb"]["centroid"])
            box_sizes.append(obj["obb"]["axesLengths"])
        return {
            "inst_labels": labels,
            "pcds": pcds,
            "obj_center": centers,
            "obj_box_size": box_sizes,
        }

    def load_scannetpp(self, scene_dir, scan_ids):
        scans = {}
        # attribute
        # inst_labels, inst_locs, inst_colors, pcds, / pcds_pred, inst_labels_pred
        for scan_id in scan_ids:
            vertices = self._load_pcd(os.path.join(scene_dir, "data", scan_id, "scans", "mesh_aligned_0.05.ply"))
            # load inst
            segments = json.load(open(os.path.join(scene_dir, "data", scan_id, "scans", "segments.json")))
            annotations = json.load(open(os.path.join(scene_dir, "data", scan_id, "scans", "segments_anno.json")))

            scans[scan_id] = self._extract_instance_annotations(vertices, segments, annotations)

        print("finish loading scannetpp data")
        return scans
