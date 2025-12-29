import os
import numpy as np
import cv2
import tensorflow as tf
from keras import layers, Model
import pandas as pd
from collections import deque

class SiameseMobileNet:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = None
    
    def create_base_network(self):
        from keras.applications import MobileNetV2
        from keras import Sequential
        
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu')
        ])
        return model
    
    def build_siamese_network(self):
        base_network = self.create_base_network()

        input_a = layers.Input(shape=self.input_shape)
        input_b = layers.Input(shape=self.input_shape)
        encoded_a = base_network(input_a)
        encoded_b = base_network(input_b)
        l1_distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded_a, encoded_b])

        prediction = layers.Dense(64, activation='relu')(l1_distance)
        prediction = layers.Dropout(0.3)(prediction)
        prediction = layers.Dense(32, activation='relu')(prediction)
        prediction = layers.Dense(1, activation='sigmoid')(prediction)

        self.model = Model(inputs=[input_a, input_b], outputs=prediction)
        return self.model

class FaceVerificationSystem:
    def __init__(self, model_path, data_path, threshold=0.5):
        self.threshold = threshold
        self.input_shape = (128, 128, 3)
        
        # Initialize Siamese Network
        print("Loading Siamese Network...")
        self.siamese_net = SiameseMobileNet(input_shape=self.input_shape)
        self.model = self.siamese_net.build_siamese_network()
        self.model.load_weights(model_path)
        print("Model loaded successfully!")
        
        # Load reference faces and image paths
        print("Loading reference faces...")
        self.reference_faces, self.reference_ids, self.reference_paths = self.load_reference_faces(data_path)
        print(f"Loaded {len(self.reference_faces)} reference faces")
        
        # Initialize face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # For display
        self.matched_image = None
        self.no_match_image = self.create_no_match_image()
        
    def load_reference_faces(self, data_path):
        """Load all reference face images from the data file"""
        df = pd.read_csv(data_path, sep='\t')
        
        reference_faces = []
        reference_ids = []
        reference_paths = {}
        
        for idx, row in df.iterrows():
            img_path = row['image_path']
            person_id = row['person_id']
            person_type = row['type']
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (self.input_shape[0], self.input_shape[1]))
                    img_normalized = img_resized.astype('float32') / 255.0
                    
                    reference_faces.append(img_normalized)
                    id_label = f"{person_type}_{person_id}"
                    reference_ids.append(id_label)
                    reference_paths[id_label] = img_path
                else:
                    print(f"Warning: Could not read image {img_path}")
            else:
                print(f"Warning: File not found {img_path}")
        
        return np.array(reference_faces), reference_ids, reference_paths
    
    def create_no_match_image(self):
        """Create a placeholder image for when no match is found"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 240
        
        # Add text
        text = "No Match"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 2, 3)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, 2, (100, 100, 100), 3)
        
        return img
    
    def preprocess_face(self, face_img):
        """Preprocess detected face for the model"""
        face_resized = cv2.resize(face_img, (self.input_shape[0], self.input_shape[1]))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype('float32') / 255.0
        return face_normalized
    
    def verify_face(self, face_img):
        """Compare face with all reference faces and return best match"""
        preprocessed_face = self.preprocess_face(face_img)
        
        # Batch process all comparisons at once for speed
        face_batch = np.repeat(np.expand_dims(preprocessed_face, axis=0), len(self.reference_faces), axis=0)
        ref_batch = self.reference_faces
        
        # Get all predictions in one forward pass
        similarities = self.model.predict([face_batch, ref_batch], verbose=0, batch_size=32).flatten()
        
        # Find the match with highest confidence
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_match_id = self.reference_ids[best_idx]
        
        # Only return the match if it exceeds threshold
        if best_similarity >= self.threshold:
            return best_match_id, best_similarity
        else:
            return "Unknown", best_similarity
    
    def load_matched_image(self, person_id):
        """Load the reference image for the matched person"""
        if person_id == "Unknown" or person_id not in self.reference_paths:
            return self.no_match_image.copy()
        
        img_path = self.reference_paths[person_id]
        img = cv2.imread(img_path)
        
        if img is not None:
            # Resize to fit the display area (480x640)
            h, w = img.shape[:2]
            target_h, target_w = 480, 640
            
            # Calculate scaling to fit within target size while maintaining aspect ratio
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # Create a canvas and center the image
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 240
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
            
            # Add label
            label = f"ID: {person_id}"
            cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            return canvas
        else:
            return self.no_match_image.copy()
    
    def run_realtime(self):
        """Run real-time face verification with GUI"""
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("\nStarting real-time face verification...")
        print("Press 'q' to quit")
        
        frame_count = 0
        last_verification = {}
        current_matched_image = self.no_match_image.copy()
        faces = []  # Initialize faces variable
        
        # FPS calculation
        import time
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS every 30 frames
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Detect faces (only every other frame)
            if frame_count % 2 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.2, 
                    minNeighbors=4, 
                    minSize=(60, 60),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            
            # Process only the largest face for better performance
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # Verify face only every 5 frames
                if frame_count % 5 == 0:
                    face_roi = frame[y:y+h, x:x+w]
                    person_id, confidence = self.verify_face(face_roi)
                    last_verification = {
                        'id': person_id,
                        'confidence': confidence,
                        'bbox': (x, y, w, h)
                    }
                    
                    # Load the matched person's image
                    current_matched_image = self.load_matched_image(person_id)
                
                # Use last verification result
                if last_verification:
                    person_id = last_verification['id']
                    confidence = last_verification['confidence']
                    
                    # Draw bounding box
                    color = (0, 255, 0) if person_id != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw label background
                    label = f"{person_id}: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x, y-30), (x+label_size[0], y), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # No face detected, show no match image
                if frame_count % 30 == 0:  # Update less frequently when no face
                    current_matched_image = self.no_match_image.copy()
            
            # Display FPS on camera frame
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Create combined display (side by side)
            combined_display = np.hstack([frame, current_matched_image])
            
            # Add labels to distinguish sections
            cv2.putText(combined_display, "Live Camera", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_display, "Matched Identity", (frame.shape[1] + 10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show combined frame
            cv2.imshow('Face Verification System', combined_display)
            
            # Quit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = './siamese_mobilenet_novel.h5'
    DATA_PATH = 'real_time_path.txt'
    THRESHOLD = 0.5  # Adjust this threshold based on your needs
    
    # Initialize and run system
    try:
        system = FaceVerificationSystem(MODEL_PATH, DATA_PATH, threshold=THRESHOLD)
        system.run_realtime()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()