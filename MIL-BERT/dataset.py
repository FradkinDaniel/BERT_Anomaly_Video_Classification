import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

# Automatically resolve the project root and data directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'DATA')

crop = 5
L2_norm = False


def process_feat_useone(feat, length):
    divided_features = []
    for f in feat:
        new_f = np.zeros((length, f.shape[1])).astype(np.float32)
        r = np.linspace(0, len(f), length + 1, dtype=int)
        for i in range(length):
            new_f[i, :] = f[r[i], :] if r[i] == r[i + 1] else f[r[i], :]
        divided_features.append(new_f)

    return np.array(divided_features, dtype=np.float32)


def process_feat(feat, length):
    divided_features = []
    for f in feat:
        new_f = np.zeros((length, f.shape[1])).astype(np.float32)
        r = np.linspace(0, len(f), length + 1, dtype=int)
        for i in range(length):
            if r[i] != r[i + 1]:
                new_f[i, :] = np.mean(f[r[i]:r[i + 1], :], axis=0)
            else:
                new_f[i, :] = f[r[i], :]
        divided_features.append(new_f)

    return np.array(divided_features, dtype=np.float32)


class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, dataset='UCF-Crime', modality='RGB+Flow', divideTo32=False, L2Norm=0, multiCrop=True,
                 path=None):
        super(Normal_Loader, self).__init__()

        self.dataset = dataset
        self.modality = modality
        self.divideTo32 = divideTo32
        self.L2Norm = L2Norm
        self.multiCrop = multiCrop

        # Determine paths based on dataset
        if dataset == 'UCF-Crime':
            # splits path and root data directory
            self.path = path or os.path.join(DATA_DIR, 'UCF-Crime', 'splits')
            self.data_root = os.path.join(DATA_DIR, 'UCF-Crime')
            train_file = 'train_001.txt'
            test_file = 'test_002.txt'
        elif dataset == 'UCF-Crime-RTFM':
            self.path = path or os.path.join(DATA_DIR, 'RTFM')
            self.data_root = self.path
            train_file = 'train.txt'
            test_file = 'test.txt'
        elif dataset == 'ShanghaiTech':
            self.path = path or os.path.join(DATA_DIR, 'ShanghaiTech')
            self.data_root = self.path
            train_file = 'train.txt'
            test_file = 'test.txt'
        elif dataset == 'XD-Violence':
            self.path = path or os.path.join(DATA_DIR, 'XD-Violence')
            self.data_root = self.path
            train_file = 'rgb_normal.list'
            test_file = 'rgb_test.list'
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.is_train = is_train
        # load split list
        list_path = os.path.join(self.path, train_file if self.is_train == 1 else test_file)
        with open(list_path, 'r') as f:
            self.data_list = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        name = self.data_list[idx]

        # XD-Violence training
        if self.dataset == 'XD-Violence' and self.is_train == 1:
            concat = []
            prefix = name.rstrip()[:-5]
            for j in range(crop):
                rgb_file = f"{prefix}{j}.npy"
                flow_file = rgb_file.replace('/RGB/', '/Flow/')
                arr = []
                if self.modality in ('RGB', 'RGB+Flow'):
                    rgb = np.load(rgb_file)
                    if self.L2Norm == 2:
                        rgb = rgb / np.linalg.norm(rgb, axis=-1, keepdims=True)
                    arr.append(rgb)
                if self.modality in ('Flow', 'RGB+Flow'):
                    flow = np.load(flow_file)
                    if self.L2Norm == 2:
                        flow = flow / np.linalg.norm(flow, axis=-1, keepdims=True)
                    arr.append(flow)
                concat.append(np.concatenate(arr, axis=1))
            feats = process_feat(np.asarray(concat), 32)
            if self.L2Norm and self.modality == 'RGB+Flow':
                half = feats.shape[-1] // 2
                feats[:, :, :half] /= np.linalg.norm(feats[:, :, :half], axis=2, keepdims=True)
                feats[:, :, half:] /= np.linalg.norm(feats[:, :, half:], axis=2, keepdims=True)
            return feats

        # UCF-Crime loading
        if self.dataset == 'UCF-Crime':
            parts = []
            if self.modality in ('RGB', 'RGB+Flow'):
                rgb_path = os.path.join(self.data_root, 'all_rgbs', f"{name}.npy")
                parts.append(np.load(rgb_path))
            if self.modality in ('Flow', 'RGB+Flow'):
                flow_path = os.path.join(self.data_root, 'all_flows', f"{name}.npy")
                parts.append(np.load(flow_path))
            features = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
            frames = features.shape[0]
            gts = []
            return features, gts, frames, name

        # UCF-Crime-RTFM loading
        if self.dataset == 'UCF-Crime-RTFM':
            base = os.path.basename(name)[:-4]
            npy = os.path.join(self.path, 'UCF_Train_ten_crop_i3d', f"{base}_i3d.npy")
            feats = np.load(npy)
            if not self.multiCrop:
                feats = feats[:, :1].transpose(1, 0, 2)
                if self.L2Norm == 2:
                    feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
                feats = process_feat(feats, 32)
                if self.L2Norm:
                    feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
                return np.squeeze(feats, 0)
            feats = feats.transpose(1, 0, 2)
            if self.L2Norm == 2:
                feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
            feats = process_feat(feats, 32)
            if self.L2Norm:
                feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
            return feats

        raise NotImplementedError(f"Unhandled dataset {self.dataset} in Normal_Loader.")


class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, dataset='UCF-Crime', modality='RGB+Flow', divideTo32=False, L2Norm=0, multiCrop=True,
                 path=None):
        super(Anomaly_Loader, self).__init__()

        self.dataset = dataset
        self.modality = modality
        self.divideTo32 = divideTo32
        self.L2Norm = L2Norm
        self.multiCrop = multiCrop

        if dataset == 'UCF-Crime':
            self.path = path or os.path.join(DATA_DIR, 'UCF-Crime', 'splits')
            self.data_root = os.path.join(DATA_DIR, 'UCF-Crime')
            train_file = 'train_001.txt'
            test_file = 'test_002.txt'
        elif dataset == 'XD-Violence':
            self.path = path or os.path.join(DATA_DIR, 'XD-Violence')
            self.data_root = self.path
            train_file = 'rgb_abnormal.list'
            test_file = 'rgb_test.list'
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.is_train = is_train
        list_file = train_file if self.is_train == 1 else test_file
        with open(os.path.join(self.path, list_file), 'r') as f:
            self.data_list = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        name = self.data_list[idx]

        if self.dataset == 'UCF-Crime':
            parts = []
            if self.modality in ('RGB', 'RGB+Flow'):
                parts.append(np.load(os.path.join(self.data_root, 'all_rgbs', f"{name}.npy")))
            if self.modality in ('Flow', 'RGB+Flow'):
                parts.append(np.load(os.path.join(self.data_root, 'all_flows', f"{name}.npy")))
            features = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
            frames = features.shape[0]
            gts = []
            return features, gts, frames, name

        if self.dataset == 'XD-Violence' and self.is_train == 1:
            concat = []
            prefix = name.rstrip()[:-5]
            for j in range(crop):
                rgb = np.load(f"{prefix}{j}.npy")
                flow = np.load(f"{prefix}{j}.npy".replace('/RGB/', '/Flow/'))
                if self.L2Norm == 2:
                    rgb /= np.linalg.norm(rgb, axis=-1, keepdims=True)
                    flow /= np.linalg.norm(flow, axis=-1, keepdims=True)
                concat.append(np.concatenate([rgb, flow], axis=1))
            feats = process_feat(np.asarray(concat), 32)
            if self.L2Norm:
                half = feats.shape[-1] // 2
                feats[:, :, :half] /= np.linalg.norm(feats[:, :, :half], axis=2, keepdims=True)
                feats[:, :, half:] /= np.linalg.norm(feats[:, :, half:], axis=2, keepdims=True)
            return feats

        raise NotImplementedError(f"Unhandled dataset {self.dataset} in Anomaly_Loader.")
