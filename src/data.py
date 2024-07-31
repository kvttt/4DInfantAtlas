import glob

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import numpy as np


class BCP(Dataset):
    """
    BCP Dataset.
    """

    def __init__(
            self,
            data_folder: str = './BCP_train',
            age_max: float = 2205.,
    ):
        """
        :param data_folder: path to the folder containing the npz files (default: './BCP_train')
                            change to './BCP_test' during inference.
        :param age_max: maximum age in the dataset, used to normalize ages (default: 2205.)
        """
        self.data_folder = data_folder
        self.age_max = age_max
        self.folders = glob.glob(data_folder + '/*')

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx: int):
        """
        :param idx: index.
        :return: rigid aligned T1WI, affine aligned T1WI, affine aligned tissue maps, age.
        """
        folder = self.folders[idx]
        rigid_t1_fn = folder + '/T1w_rigid.npz'
        affine_t1_fn = folder + '/T1w_affine.npz'
        affine_seg_fn = folder + '/T1w_affine_seg.npz'

        age = float(folder.split('/')[-1].split('_')[-1]) / self.age_max  # assume folder name to be 'XXBCP######_###'
        age = torch.from_numpy(np.array([age])).float()

        rigid_t1 = np.load(rigid_t1_fn)['arr_0']  # assume shape (1, 160, 192, 160)
        rigid_t1 = torch.from_numpy(rigid_t1).float()
        affine_t1 = np.load(affine_t1_fn)['arr_0']  # assume shape (1, 160, 192, 160)
        affine_t1 = torch.from_numpy(affine_t1).float()
        affine_seg = np.load(affine_seg_fn)['arr_0']  # assume shape (3, 160, 192, 160)
        affine_seg = torch.from_numpy(affine_seg).float()

        return rigid_t1, affine_t1, affine_seg, age


def create_weighted_sampler_bcp(data_folder: str = './BCP_train'):
    """
    Use the ages to create a weighted sampler.

    :param data_folder: folder containing the npz files.
    :return: an instance of torch.utils.data.WeightedRandomSampler.
    """

    frequencies = []
    folders = glob.glob(data_folder + '/*')
    for folder in folders:
        age = float(folder.split('/')[-1].split('_')[-1])  # in days
        frequencies.append(age / 91.25)  # in 3 months
    frequencies = np.array(frequencies).round().astype(int)

    ages, counts = np.unique(frequencies, return_counts=True)
    weights = 1. / counts  # (https://discuss.pytorch.org/t/how-to-implement-oversampling-in-cifar-10/16964/6)
    samples_weights = weights[frequencies]
    samples_weights = torch.from_numpy(samples_weights)

    return WeightedRandomSampler(samples_weights, len(samples_weights))


def create_rough_template(data_folder: str = '../BCP_train'):
    """
    Creates a rough template from the images.

    :param data_folder: folder containing the npz files.
    :return: None.
    """

    folders = glob.glob(data_folder + '/*')
    count = 0
    t1 = np.zeros((1, 160, 192, 160))
    seg = np.zeros((3, 160, 192, 160))

    for folder in folders:
        affine_t1_fn = folder + '/T1w_affine.npz'
        affine_seg_fn = folder + '/T1w_affine_seg.npz'
        count += 1
        t1 += np.load(affine_t1_fn)['arr_0']
        seg += np.load(affine_seg_fn)['arr_0']

    t1 /= count
    seg /= count

    np.savez_compressed('../rough_template_t1.npz', t1)
    np.savez_compressed('../rough_template_seg.npz', seg)
