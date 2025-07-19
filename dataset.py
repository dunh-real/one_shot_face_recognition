import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform = None):
        self.triplets = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path = os.path.join(self.img_dir, self.triplets.iloc[idx]['anchor'])
        positive_path = os.path.join(self.img_dir, self.triplets.iloc[idx]['pos'])
        negative_path = os.path.join(self.img_dir, self.triplets.iloc[idx]['neg'])

        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img
    

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])