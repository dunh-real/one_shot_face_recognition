import cv2
import torch
import numpy as np
from PIL import Image
import  torch.nn.functional as F
from torchvision import transforms
from model import Siamese

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = Siamese(embedding_dim = 128).to(device)
model.load_state_dict(torch.load('siamese_model.pth'))
model.eval()

