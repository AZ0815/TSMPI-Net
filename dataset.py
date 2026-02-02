import os
import h5py
import numpy as np
import torch
from torch.utils import data

class H5Dataset(data.Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(self.folder_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data_1d_seq_ready = []
        data_2d_seq_ready = []

        dir_path = os.path.join(self.folder_path, self.file_list[idx])
        file_path_list = [q for q in os.listdir(dir_path) if q.endswith('.h5')]
        file_path_list.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))

        for i in file_path_list:
            file_path = os.path.join(dir_path, i)
            with h5py.File(file_path, 'r') as f:
                signal_idx = np.random.randint(1, 11)
                signal_key = f'signal{signal_idx}'
                data_1d = np.array(f[signal_key])
                data_2d = np.array(f['image'])
                data_2d = data_2d / np.max(data_2d)

            data_1d = torch.tensor(data_1d, dtype=torch.float32)
            data_2d = torch.tensor(data_2d, dtype=torch.float32)
            data_2d = data_2d.unsqueeze(0).unsqueeze(0)

            data_1d_seq_ready.append(data_1d)
            data_2d_seq_ready.append(data_2d)

        data_1d_seq = torch.cat(data_1d_seq_ready[0:], dim=0)
        data_2d_seq = torch.cat(data_2d_seq_ready[0:], dim=1)

        return data_1d_seq, data_2d_seq


class CleanH5Dataset(data.Dataset):
    def __init__(self, base: H5Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x1d, x2d = self.base[idx]
        x1d = torch.nan_to_num(x1d, nan=0.0, posinf=0.0, neginf=0.0)
        x2d = torch.nan_to_num(x2d, nan=0.0, posinf=0.0, neginf=0.0)
        return x1d, x2d

__all__ = ["H5Dataset", "CleanH5Dataset"]
