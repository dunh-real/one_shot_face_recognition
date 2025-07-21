import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from facenet_pytorch import InceptionResnetV1
import os

# Configuration
DATA_DIR = "./subset_data"  # Path to preprocessed data
MODEL_PATH = "siamese_facenet.pt"  # Path to save trained model
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TripletDataset(Dataset):
    """Custom Dataset for loading triplets (anchor, positive, negative)."""
    def __init__(self, anchors, positives, negatives):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor = torch.tensor(self.anchors[idx], dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        positive = torch.tensor(self.positives[idx], dtype=torch.float32).permute(2, 0, 1)
        negative = torch.tensor(self.negatives[idx], dtype=torch.float32).permute(2, 0, 1)
        return anchor, positive, negative

def triplet_loss(anchor, positive, negative, margin=1.0):
    """Compute triplet loss."""
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def evaluate_model(model, test_images, test_ids, device):
    """Evaluate model accuracy on test set."""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for img, id_ in zip(test_images, test_ids):
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            emb = model(img).cpu().numpy()
            embeddings.append(emb[0])
            labels.append(id_)
    
    embeddings = np.array(embeddings)
    correct = 0
    for i, (emb, true_id) in enumerate(zip(embeddings, labels)):
        distances = np.sum((embeddings - emb) ** 2, axis=1)
        pred_idx = np.argmin(distances)
        pred_id = labels[pred_idx]
        if pred_id == true_id:
            correct += 1
    
    accuracy = correct / len(test_images) * 100
    return accuracy

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_anchors = np.load(os.path.join(DATA_DIR, "train_anchors.npy"))
    train_positives = np.load(os.path.join(DATA_DIR, "train_positives.npy"))
    train_negatives = np.load(os.path.join(DATA_DIR, "train_negatives.npy"))
    test_images = np.load(os.path.join(DATA_DIR, "test_images.npy"))
    test_ids = np.load(os.path.join(DATA_DIR, "test_ids.npy"))
    print(f"Loaded {len(train_anchors)} training triplets, {len(test_images)} test images")

    # Create dataset and dataloader
    dataset = TripletDataset(train_anchors, train_positives, train_negatives)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    model.train()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            
            optimizer.zero_grad()
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        accuracy = evaluate_model(model, test_images, test_ids, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}, Test Accuracy: {accuracy:.2f}%")
        
        # Save model checkpoint
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved model checkpoint to {MODEL_PATH}")

    print("Training completed.")

if __name__ == "__main__":
    main()