import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from time import time

# Configuration
MODEL_PATH = "siamese_facenet.pt"  # Path to trained model
DATA_DIR = "./subset_data"  # Path to preprocessed data
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (160, 160)  # FaceNet input size
THRESHOLD = 0.6  # Similarity threshold for matching
FPS_TARGET = 15  # Target frames per second

def preprocess_image(img):
    """Preprocess image: resize to 160x160, normalize to [0,1], convert to tensor."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img

def create_employee_database(model, test_images, test_ids):
    """Create database of embeddings for test images."""
    model.eval()
    embeddings = []
    ids = []
    with torch.no_grad():
        for img, id_ in zip(test_images, test_ids):
            img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            emb = model(img_tensor).cpu().numpy()[0]
            embeddings.append(emb)
            ids.append(id_)
    return np.array(embeddings), np.array(ids)

def main():
    # Initialize model and MTCNN
    model = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)

    # Load test data for employee database
    test_images = np.load(os.path.join(DATA_DIR, "test_images.npy"))
    test_ids = np.load(os.path.join(DATA_DIR, "test_ids.npy"))
    embeddings, ids = create_employee_database(model, test_images, test_ids)
    print(f"Loaded employee database with {len(embeddings)} embeddings")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Could not open webcam")

    frame_count = 0
    start_time = time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame_count += 1
        # Process every 2nd frame to achieve ~15 FPS
        if frame_count % 2 == 0:
            # Detect faces
            boxes, _ = mtcnn.detect(frame)
            result = 0  # Default: unrecognized
            label = "Unrecognized (0)"

            if boxes is not None:
                for box in boxes:
                    # Extract face
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Generate embedding
                    try:
                        face_tensor = preprocess_image(face)
                        with torch.no_grad():
                            embedding = model(face_tensor).cpu().numpy()[0]

                        # Compare with database
                        distances = np.sum((embeddings - embedding) ** 2, axis=1)
                        min_distance = np.min(distances)
                        if min_distance < THRESHOLD:
                            result = 1
                            label = f"Recognized (1)"

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

            # Print result to console
            print(f"Frame {frame_count}: {label}")

        # Display frame
        cv2.imshow("Face Recognition", frame)

        # Control FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Measure FPS
        elapsed_time = time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()