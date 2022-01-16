import random
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset

class DatasetProcessing(Dataset):

    def __init__(self, path_images, path_samples, transform=None):

        # Set random seed for reproducibility of data split
        random.seed(42)

        self.path_images  = path_images     # path to directory with dataset images
        self.path_samples = path_samples    # path to a file listing selected samples
        self.transform    = transform       # transform to process the images

        self.image_ids        = []
        self.severity_classes = []
        self.lesions_count    = []

        samples_file = open(path_samples, 'r')

        # Read each sample and populate the corresponding image, severity class, and lesion count
        for line in samples_file.readlines():
            filename, label, lesion = line.split()
            self.image_ids.append(filename)
            self.severity_classes.append(int(label))
            self.lesions_count.append(int(lesion))

        samples_file.close()

        # Convert collected data into arrays
        self.image_ids       = np.array(self.image_ids)
        self.severity_classes = np.array(self.severity_classes)
        self.lesions_count  = np.array(self.lesions_count)

        # Shuffle the samples, keeping them grouped by severity class ([Rotem] in fact, why keep them grouped?)
        if 'NNEW_trainval' in path_samples:
            shuffled_indexes = []
            for i in range(4):
                severity_i = list(np.where(self.severity_classes == i)[0])
                shuffled_i = random.sample(severity_i, len(severity_i))
                shuffled_indexes.extend(shuffled_i)
            self.image_ids       = self.image_ids[shuffled_indexes]
            self.severity_classes = self.severity_classes[shuffled_indexes]
            self.lesions_count  = self.lesions_count[shuffled_indexes]


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path_images, self.image_ids[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(np.array(self.severity_classes[index]))
        lesion = torch.from_numpy(np.array(self.lesions_count[index]))
        return img, label, lesion


    def __len__(self):
        return len(self.image_ids)

