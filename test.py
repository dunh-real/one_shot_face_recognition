import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Siamese
import torch.nn.functional as F
from torchvision import transforms

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load test dataset
test_dataset = Dataset(
    csv_file = 'dataset/triplets.csv',
    img_dir = 'dataset/images',
    transform = transform
)

test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

# Load model
model = Siamese(embedding_dim = 128)
model.load_state_dict(torch.load('siamese_model.pth'))
model.to(device)
model.eval()

# Evaluation
correct = 0
total = 0
threshold = 0.5 # adjust based on validation

with torch.no_grad():
    for anchor, positive, negative in test_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        anchor_emb, pos_emb, neg_emb = model.forward_triplet(anchor, positive, negative)

        # compute distances
        pos_distance = F.pairwise_distance(anchor_emb, pos_emb)
        neg_distance = F.pairwise_distance(anchor_emb, neg_emb)

        # verification: positive distance should be smaller than negative distance
        correct += ((pos_distance < neg_distance) & (pos_distance < threshold)).sum().item()
        total += anchor.size(0)

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")