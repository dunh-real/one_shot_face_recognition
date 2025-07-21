import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Configuration
dataset_dir = './dataset/images'
csv_file = './dataset/triplets.csv'
num_individuals = 300
img_size = (160, 160)
output_dir = 'subset_data'

def load_and_preprocess_img(img_path):
    # load and preprocess an image: resize to 160x160, normalize to [0, 1]
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    return img

def main():
    # create output directory
    os.makedirs(output_dir, exist_ok = True)

    # load CSV
    print("Loading CSV...")
    df = pd.read_csv(csv_file)

    # filter first 300 individuals based on id1
    unique_ids = df['id1'].unique()[:num_individuals]
    df_subset = df[df['id1'].isin(unique_ids)]
    print(f"Selected {len(df_subset)} triplets for {num_individuals} individuals")

    # split into train (80%) and test (20%) by individuals
    train_ids, test_ids = train_test_split(unique_ids, test_size = 0.2, random_state = 42)
    train_df = df_subset[df_subset['id1'].isin(train_ids)]
    test_df = df_subset[df_subset['id1'].isin(test_ids)]
    print(f"Training set: {len(train_df)} triplets, Test set: {len(test_df)} triplets")

    # preprocess training triplets
    train_triplets = []
    for _, row in train_df.iterrows():
        try:
            anchor_path = os.path.join(dataset_dir, row['anchor'])
            pos_path = os.path.join(dataset_dir, row['pos'])
            neg_path = os.path.join(dataset_dir, row['neg'])

            anchor_img = load_and_preprocess_img(anchor_path)
            pos_img = load_and_preprocess_img(pos_path)
            neg_img = load_and_preprocess_img(neg_path)

            train_triplets.append({
                'anchor': anchor_img,
                'positive': pos_img,
                'negative': neg_img,
                'id': row['id1']
            })
        except Exception as e:
            print(f"Error processing triplet for ID {row['id1']}: {e}")
            continue
    
    # preprocess test images (anchor only, for evaluation)
    test_images = []
    test_ids_list = []
    for _, row in test_df.iterrows():
        try:
            anchor_path = os.path.join(dataset_dir, row['anchor'])
            anchor_img = load_and_preprocess_img(anchor_path)
            test_images.append(anchor_img)
            test_ids_list.append(row['id1'])
        except Exception as e:
            print(f"Error processing test image for ID {row['id1']}: {e}")
            continue

    # convert to numpy arrays
    train_anchors = np.array([t['anchor'] for t in train_triplets])
    train_positives = np.array([t['positive'] for t in train_triplets])
    train_negatives = np.array([t['negative'] for t in train_triplets])
    train_ids = np.array([t['id'] for t in train_triplets])
    test_images = np.array(test_images)
    test_ids = np.array(test_ids_list)

    # save preprocessed data
    np.save(os.path.join(output_dir, 'train_anchors.npy'), train_anchors)
    np.save(os.path.join(output_dir, 'train_positives.npy'), train_positives)
    np.save(os.path.join(output_dir, 'train_negatives.npy'), train_negatives)
    np.save(os.path.join(output_dir, 'train_ids.npy'), train_ids)
    np.save(os.path.join(output_dir, 'test_images.npy'), test_images)
    np.save(os.path.join(output_dir, 'test_ids.npy'), test_ids)

    print(f"Saved preprocessed data to {output_dir}")
    print(f"Training triplets: {len(train_triplets)}, Test images: {len(test_images)}")

if __name__ == "__main__":
    main()