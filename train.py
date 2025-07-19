import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Siamese
import torch.nn as nn
from torchvision import transforms

class TripletLoss(nn.Module):
    def __init__(self, margin = 0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        lossess = torch.relu(distance_positive - distance_negative + self.margin)
        return lossess.mean()
    
# hyperparameters
batch_size = 64
learning_rate = 0.0001
num_epochs = 1
embedding_dim = 128
margin = 0.2

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset and dataloader
dataset = Dataset(csv_file='dataset/triplets.csv',
                  img_dir='dataset/images',
                  transform=transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initial model, loss, and optimize
model = Siamese(embedding_dim=embedding_dim).to(device)
criterion = TripletLoss(margin=margin)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # forward pass
        anchor_emb, pos_emb, neg_emb = model.forward_triplet(anchor, positive, negative)
        loss = criterion(anchor_emb, pos_emb, neg_emb)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"[{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average loss {total_loss/len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), "siamese_model.pth")