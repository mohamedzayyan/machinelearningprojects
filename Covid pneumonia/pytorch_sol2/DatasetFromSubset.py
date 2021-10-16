import torch
from torch.utils.data import Dataset, TensorDataset, random_split
from torchvision import transforms

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        #self.subset.indices = range(0,len(self.subset.indices))
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __getitem__(self, idx):
        img_id = self.subset[idx]['id']
        labels = self.subset[idx]['labels']
        image = self.subset[idx]['image']
        if self.transform:
            sample = {'image': self.transform(image), 'labels': labels, 'id': img_id}
        else:
            sample = {'image': image, 'labels': labels, 'id': img_id}
        return sample

    def __len__(self):
        return len(self.subset)