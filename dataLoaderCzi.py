from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import argparse
import warnings
warnings.filterwarnings("ignore")
from czireader import CziReader
import scipy
from bufferedpatchdataset import BufferedPatchDataset
from PIL import Image

def normalize(img):
    """Subtract mean, set STD to 1.0"""
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result

class Resizer(object):
    def __init__(self, factors):
        """
        factors - tuple of resizing factors for each dimension of the input array"""
        self.factors = factors

    def __call__(self, x):
        return scipy.ndimage.zoom(x, (self.factors), mode='nearest')

    def __repr__(self):
        return 'Resizer({:s})'.format(str(self.factors))


# class for dataset
class CZIdataset(Dataset):
    def __init__(self, csvFile, transform=None):
        self.ds = pd.read_csv(csvFile)        
        self.transform = transform 
        assert all(i in self.ds.columns for i in ['path_czi', 'channel_signal', 'channel_target']) 
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        element = self.ds.iloc[index, :]
        has_target = not np.isnan(element['channel_target'])
        czi = CziReader(element['path_czi'])
        
        im_out = list()
        im_out.append(czi.get_volume(element['channel_signal']))
        if has_target:
            im_out.append(czi.get_volume(element['channel_target']))
        
        if self.transform is not None:
            for t in self.transform: 
                im_out[0] = eval(t)(im_out[0])

        im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]
        
        #unsqueeze to make the first dimension be the channel dimension
        print(im_out[0].shape)
        im_out = [torch.unsqueeze(im, 0) for im in im_out]
        print(im_out[0].shape)
        return im_out        

if __name__ == "__main__":
    factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
    default_resizer_str = 'Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--transform', nargs='+', default=['normalize', default_resizer_str], help='list of transforms on Dataset signal')
    opts = parser.parse_args()
    
    transformed_dataset = CZIdataset(csvFile=opts.path_dataset_csv, 
                            transform=opts.transform)

    ds_patch = BufferedPatchDataset(
        dataset = transformed_dataset,
        patch_size = [32, 64, 64],
        buffer_size = 1,
        buffer_switch_frequency = 720,
        npatches = 6,
        verbose = True,
        shuffle_images = True,
        **{},
    )
    dataloader = DataLoader(
        ds_patch,
        batch_size = 1,
    )

    for j, sample in enumerate(dataloader):
        print(sample[0].shape)
        
        count = 1
        for  idx in range(1):
            imin_all = sample[0][idx, 0, :]
            imout_all = sample[1][idx, 0, :]

            randomidx = np.random.randint(0, 32, 4)
            
            for i in randomidx:
                plt.figure(count)
                count += 1

                imin = imin_all[i] 
                imout = imout_all[i]

                ax = plt.subplot(1, 2, 1)
                plt.tight_layout()
                ax.set_title('sample #{:d}, slice #{:d}, input'.format(idx, i))
                ax.axis('off')
                plt.imshow(imin)

                ax = plt.subplot(1, 2, 2)                
                plt.tight_layout()
                ax.set_title('sample #{:d}, slice #{:d}, target'.format(idx, i))
                ax.axis('off')
                plt.imshow(imout)

    plt.show()