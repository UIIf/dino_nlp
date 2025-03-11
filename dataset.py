from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os


class TextFolderDataset(Dataset):
    def __init__(self, dataset_path, transforms):
        super().__init__()
        self.paths = self._get_all_files(dataset_path)
        self.transforms = transforms

    def _get_all_files(self, dataset_path):
        origins = [dataset_path]
        paths = []

        while len(origins) > 0:
            dataset_path = origins.pop()
            for file in os.listdir(dataset_path):
                path = os.path.join(dataset_path, file)
                if os.path.isdir(path):
                    origins.append(path)
                else:
                    if path.split(".")[-1] == "c":
                        paths.append(path)
        return paths

    def __getitem__(self, idx):
        try:
            with open(self.paths[idx], "r") as f:
                return self.transforms(f.read())
        except FileNotFoundError:
            raise ValueError(f"Problem with file {self.paths[idx]}")

    def __len__(self):
        return len(self.paths)