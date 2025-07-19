import torch
import torch.nn as nn
import torchvision.models as models

class Siamese(nn.Module):
    def __init__(self, embedding_dim = 128):
        super(Siamese, self).__init__()
        # load pre-trained ResNet18
        self.cnn = models.resnet18(pretrained=True)
        # replace the final fully connected layer
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embedding_dim)
    
    def forward(self, x):
        return self.cnn(x)
    
    def forward_triplet(self, anchor, positive, negative):
        anchor_embedding = self(anchor)
        positive_embedding = self(positive)
        negative_embedding = self(negative)
        return anchor_embedding, positive_embedding, negative_embedding
    
    