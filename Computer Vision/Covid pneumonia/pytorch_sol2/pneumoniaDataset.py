import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
class pneumoniaDataset(Dataset):
	def __init__(self, root_dir, transform=None):
	    self.imgs_path = root_dir
	    file_list = glob.glob(self.imgs_path + "*")
	    #print(file_list)
	    self.data = []
	    for class_path in file_list:
	        class_name = class_path.split("/")[-1]
	        for img_path in glob.glob(class_path + "/*.jpeg"):
	            self.data.append([img_path, class_name])
	    #print(self.data)
	    self.class_map = {"normal" : 0, "opacity": 1}
	    self.transform = transform
	    
	def __len__(self):
	    return len(self.data)
	def __getitem__(self, idx):
	    img_path, class_name = self.data[idx]
	    img = cv2.imread(img_path)
	    class_id = self.class_map[class_name]
	    #img_tensor = torch.from_numpy(img)
	    class_id = torch.tensor([class_id])
	    if self.transform:
	    	sample = {'image': self.transform(img), 'labels': class_id}
	    else:
	    	sample = {'image': torch.from_numpy(img), 'labels': class_id}
	    return sample