from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()

# load the csv file containing the data
landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 17
img_name = landmarks_frame.iloc[n]['image_name']
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)
print(landmarks.shape)

def show_landmarks(image, landmarks):
    print(image.shape)
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)

plt.figure()
show_landmarks(io.imread(os.path.join('faces/', img_name)), landmarks)
plt.show(block=True)

# class for dataset
class FaceLandmarksDataset(Dataset):
    def __init__(self, csvFile, rootDir, transform=None):
        self.ds = pd.read_csv(csvFile)
        self.rootDir = rootDir
        self.transform = transform  
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.rootDir, self.ds.iloc[index]['image_name'])
        img = io.imread(img_name)
        landmarks = self.ds.iloc[index, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)

        sample = {'image': img, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
face_ds = FaceLandmarksDataset(csvFile='faces/face_landmarks.csv', rootDir='faces/')

fig = plt.figure()

for i in range(len(face_ds)):
    sample = face_ds[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show(block=True)
        break


class RescaleImage(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                newh, neww = h*self.output_size/w, self.output_size
            else:
                newh, neww = self.output_size, w*self.output_size/h
        else:
            newh, neww = self.output_size
        
        newh, neww = int(newh), int(neww)

        img = transform.resize(image, (newh, neww))

        landmarks = landmarks * [neww/w, newh/h]

        return {'image': img, 'landmarks':landmarks}


class CropImage(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        newh, neww = self.output_size

        top = np.random.randint(0, h - newh)
        left = np.random.randint(0, w - neww)

        img = image[top:top+newh, left:left+neww]
        landmarks = landmarks - [left, top]

        return {'image': img, 'landmarks': landmarks}


class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}

scale = RescaleImage(256)
crop = CropImage(128)
compose = transforms.Compose([RescaleImage(256), CropImage(224)])

fig = plt.figure()
sample = face_ds[65]

for i, trsfm in enumerate([scale, crop, compose]):
    transformed_sample = trsfm(sample)

    ax = plt.subplot(1, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(trsfm).__name__)
    show_landmarks(**transformed_sample)

plt.show(block=True)

transformed_dataset = FaceLandmarksDataset(csvFile='faces/face_landmarks.csv', 
                            rootDir='faces/', 
                            transform=transforms.Compose([RescaleImage(256), CropImage(224), ToTensor()]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break


dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=1)
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i, sample_batched in enumerate(dataloader):
    print(i, sample_batched['image'].size(), sample_batched['landmarks'].size())
    if i == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
