import os
from sklearn.datasets import load_svmlight_file

import torch
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, TensorDataset
import torch.nn.functional as F

DOWNLOAD_LINKS = {
    'mushrooms': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms',
    'ijcnn1': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2',
    'w8a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a',
    'a9a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a',
    'phishing': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing',
}


class LibSVM(Dataset):
    """
    Adaptation of LibSVM dataset to standard PyTorch Dataset
    """
    def __init__(self, root, dataset_name, download=True):
        download_link = DOWNLOAD_LINKS[dataset_name]
        target = os.path.basename(download_link)
        file_dir = os.path.join(root, target)
        if not os.path.exists(file_dir):
            if download:
                download_url(download_link, root)
            else:
                raise FileNotFoundError(f"{root} does not exist and `download is set to `False`")

        data = load_svmlight_file(file_dir)
        x = torch.from_numpy(data[0].toarray()).float()
        # normalize to norm 1 rows
        x = F.normalize(x, p=2, dim=1)
        # 0 and 1 as labels
        y = torch.tensor(data[1])
        y = (y == torch.max(y)).float().reshape(-1, 1)
        self.dataset = TensorDataset(x, y)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return self.dataset.__len__()
