import cv2
from gtts import gTTS
import pygame
from ultralytics import YOLO
from deepface import DeepFace
import time
import numpy as np
import threading
import os
import tempfile
import pickle
from pathlib import Path
import json
import platform
from config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    UNKNOWN_GREETING,
    FACE_MODEL,
    PERSON_DETECTION_CONFIDENCE,
    GREETING_COOLDOWN_SECONDS,
    MOVEMENT_THRESHOLD_PIXELS,
    CAMERA_TYPE,
    PICAMERA_RESOLUTION,
    PICAMERA_FRAMERATE,
    ENABLE_PI_OPTIMIZATIONS,
    PI_FRAME_SKIP
)

# Try to import picamera2 for Raspberry Pi support
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


def is_raspberry_pi():
    """Check if running on Raspberry Pi"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'Raspberry Pi' in line:
                    return True
    except:
        pass
    return False


def load_person_config(person_folder):
    """Load configuration for a person from their config file"""
    config_file = Path(person_folder) / "config.txt"
    
    config = {
        "name": Path(person_folder).name.title(),  # Default to folder name
        "greeting": f"Hello {Path(person_folder).name.title()}!",  # Default greeting
        "threshold": None  # Use default threshold
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key == 'name':
                            config['name'] = value
                        elif key == 'greeting':
                            config['greeting'] = value
                        elif key == 'threshold':
                            try:
                                config['threshold'] = float(value)
                            except ValueError:
                                pass
        except Exception as e:
            print(f"⚠️  Error reading config for {person_folder}: {e}")
    
    return config


def discover_people():
    """Discover all people by scanning the faces/ directory"""
    faces_dir = Path("faces")
    if not faces_dir.exists():
        return []
    
    people = []
    for person_folder in sorted(faces_dir.iterdir()):
        if not person_folder.is_dir():
            continue
        
        # Skip hidden folders and examples
        if person_folder.name.startswith('.') or person_folder.name == 'examples':
            continue
        
        # Check if folder has any image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(person_folder.glob(ext))
        
        if not image_files:
            continue
        
        # Load config for this person
        config = load_person_config(person_folder)
        
        people.append({
            "name": config["name"],
            "greeting": config["greeting"],
            "face_folder": str(person_folder),
            "threshold": config["threshold"]
        })
    
    return people


class PersonDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        pygame.mixer.init()
        self.last_person_detected = None
        self.last_description_time = 0
        self.description_cooldown = GREETING_COOLDOWN_SECONDS
        self.is_speaking = False
        self.lock = threading.Lock()
        self.embeddings_db = {}  # Store face embeddings for each person
        self.known_people = []  # List of known people
        
        print("🚀 Initializing Office Greeter with Face Recognition...")
        print("="*60)
        
        # Discover people from faces/ directory
        self.known_people = discover_people()
        
        if not self.known_people:
            print("❌ No people found in faces/ directory!")
            print("\n   To add people:")
            print("   1. Create a folder: faces/yourname/")
            print("   2. Add photos: faces/yourname/photo1.jpg")
            print("   3. Create config: faces/yourname/config.txt")
            print("   4. See faces/person.config.example for format")
            return
        
        print(f"📋 Discovered {len(self.known_people)} people:")
        for person in self.known_people:
            threshold_info = f" (custom threshold: {person['threshold']})" if person['threshold'] else ""
            print(f"   - {person['name']}{threshold_info}")
        
        # Load or generate embeddings
        self.load_face_embeddings()
        
    def load_face_embeddings(self):
        """Load or generate face embeddings for all known people"""
        embeddings_file = "face_embeddings.pkl"
        
        # Try to load cached embeddings
        if os.path.exists(embeddings_file):
            print("📂 Loading cached face embeddings...")
            try:
                with open(embeddings_file, 'rb') as f:
                    self.embeddings_db = pickle.load(f)
                print(f"✅ Loaded embeddings for {len(self.embeddings_db)} people")
                for name in self.embeddings_db.keys():
                    print(f"   - {name}: {len(self.embeddings_db[name])} face(s)")
                return
            except Exception as e:
                print(f"⚠️  Error loading cached embeddings: {e}")
                print("   Regenerating embeddings...")
        
        # Generate new embeddings
        print("\n🔍 Generating face embeddings from reference photos...")
        print("   (This may take a minute on first run...)")
        
        for person in self.known_people:
            name = person["name"]
            face_folder = person["face_folder"]
            
            # Find all image files in the folder
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(Path(face_folder).glob(ext))
            
            # Generate embeddings for each photo
            person_embeddings = []
            print(f"\n   Processing {name}...")
            
            for img_path in image_files:
                try:
                    print(f"      - {img_path.name}...", end=" ")
                    
                    # Generate embedding using DeepFace
                    embedding_objs = DeepFace.represent(
                        img_path=str(img_path),
                        model_name=FACE_MODEL,
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    # DeepFace.represent returns a list of embeddings (one per face)
                    if embedding_objs and len(embedding_objs) > 0:
                        embedding = embedding_objs[0]["embedding"]
                        person_embeddings.append(embedding)
                        print("✓")
                    else:
                        print("✗ (no face detected)")
                        
                except Exception as e:
                    print(f"✗ (error: {e})")
            
            if person_embeddings:
                self.embeddings_db[name] = person_embeddings
                print(f"   ✅ Loaded {len(person_embeddings)} face(s) for {name}")
            else:
                print(f"   ❌ No valid faces found for {name}")
        
        # Save embeddings to cache
        if self.embeddings_db:
            try:
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(self.embeddings_db, f)
                print(f"\n💾 Saved embeddings to {embeddings_file}")
            except Exception as e:
                print(f"⚠️  Could not save embeddings cache: {e}")
        
        print("\n" + "="*60)
        if not self.embeddings_db:
            print("❌ ERROR: No face embeddings loaded!")
            print("   Please add photos to the faces/ folders and restart.")
            print("\n   Expected structure:")
            print("   faces/")
            print("   ├── alexander/photo1.jpg")
            print("   ├── beatrix/photo1.jpg")
            print("   ├── claus/photo1.jpg")
            print("   └── benno/photo1.jpg")
            return
        
        print(f"✅ Ready to recognize {len(self.embeddings_db)} people!")
        print("="*60 + "\n")
    
    def find_matching_person(self, face_embedding):
        """Find the best matching person based on face embedding"""
        if not face_embedding or not self.embeddings_db:
            return None, 0.0
        
        best_match = None
        best_distance = float('inf')
        
        # Compare against all known people
        for person in self.known_people:
            name = person["name"]
            
            if name not in self.embeddings_db:
                continue
            
            # Compare against all stored embeddings for this person
            for stored_embedding in self.embeddings_db[name]:
                # Calculate cosine distance
                distance = self.cosine_distance(face_embedding, stored_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = person
        
        # Use person-specific threshold or default
        threshold = best_match["threshold"] if best_match and best_match["threshold"] else DEFAULT_SIMILARITY_THRESHOLD
        
        # Return match only if distance is below threshold
        if best_distance < threshold:
            return best_match, best_distance
        else:
            return None, best_distance
    
    def cosine_distance(self, embedding1, embedding2):
        """Calculate cosine distance between two embeddings"""
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        # Convert to distance (0 = identical, 1 = completely different)
        distance = 1 - similarity
        return distance
    
    def greet_person(self, bbox, frame):
        """Identify and greet the person using face embeddings"""
        x1, y1, x2, y2 = bbox
        
        # Extract person crop
        person_crop = frame[y1:y2, x1:x2]
        
        print("\n" + "="*60)
        print("🔍 Analyzing face...")
        
        try:
            # Generate embedding for detected face
            embedding_objs = DeepFace.represent(
                img_path=person_crop,
                model_name=FACE_MODEL,
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if not embedding_objs or len(embedding_objs) == 0:
                print("❌ No face detected in frame")
                return
            
            face_embedding = embedding_objs[0]["embedding"]
            
            # Find matching person
            matched_person, distance = self.find_matching_person(face_embedding)
            
            # Display results
            print("\n📊 Recognition Results:")
            for person in self.known_people:
                name = person["name"]
                if name not in self.embeddings_db:
                    continue
                
                # Calculate distance to this person
                min_dist = float('inf')
                for stored_emb in self.embeddings_db[name]:
                    dist = self.cosine_distance(face_embedding, stored_emb)
                    min_dist = min(min_dist, dist)
                
                match_pct = max(0, (1 - min_dist) * 100)
                status = "✓" if matched_person and matched_person["name"] == name else " "
                threshold = person["threshold"] if person["threshold"] else DEFAULT_SIMILARITY_THRESHOLD
                threshold_marker = f" (threshold: {threshold})" if name == (matched_person["name"] if matched_person else None) else ""
                print(f"   {status} {name}: {match_pct:.1f}% match (distance: {min_dist:.3f}){threshold_marker}")
            
            if matched_person:
                greeting = matched_person["greeting"]
                person_name = matched_person["name"]
                confidence = max(0, (1 - distance) * 100)
                print(f"\n✅ RECOGNIZED: {person_name} ({confidence:.1f}% confidence)")
            else:
                greeting = UNKNOWN_GREETING
                person_name = "Unknown"
                best_threshold = DEFAULT_SIMILARITY_THRESHOLD
                print(f"\n⚠️  UNKNOWN PERSON (best match: {(1-distance)*100:.1f}%, threshold: {(1-best_threshold)*100:.1f}%)")
            
            print(f"🗣️  Speaking: {greeting}")
            print("="*60)
            
            # Speak the greeting
            with self.lock:
                if not self.is_speaking:
                    self.is_speaking = True
                    threading.Thread(target=self._speak, args=(greeting,), daemon=True).start()
                    
        except Exception as e:
            print(f"❌ Error during recognition: {e}")
            import traceback
            traceback.print_exc()
    
    def _speak(self, text):
        """Speak text using gTTS with better voice quality"""
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_file)
            
            # Play audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            os.unlink(temp_file)
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            with self.lock:
                self.is_speaking = False
    
    def should_greet(self, bbox):
        """Check if enough time has passed and person has moved significantly"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_description_time < self.description_cooldown:
            return False
        
        # Check if person has moved significantly or is new
        if self.last_person_detected is not None:
            x1, y1, x2, y2 = bbox
            lx1, ly1, lx2, ly2 = self.last_person_detected
            
            # Calculate how much the person has moved
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            last_center_x = (lx1 + lx2) / 2
            last_center_y = (ly1 + ly2) / 2
            
            movement = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
            
            # Only greet if moved significantly (>150 pixels) - indicates new entry
            if movement < 150:
                return False
        
        return True
        
    def run(self):
        """Main loop for webcam detection"""
        if not self.embeddings_db:
            print("❌ Cannot start: No face embeddings loaded!")
            print("   Add photos to faces/ folders and restart.")
            return
        
        if CAMERA_TYPE == "picamera" and PICAMERA_AVAILABLE and is_raspberry_pi():
            print("🎥 Using Raspberry Pi Camera with picamera2...")
            picam2 = Picamera2()
            picam2.configure(picam2.create_preview_configuration(
                main={"size": PICAMERA_RESOLUTION, "format": "RGB888"}))
            picam2.start()
        else:
            print("🎥 Using default webcam...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return
        
        print("🎥 Starting detection...")
        print(f"⏱️  Greetings cooldown: {self.description_cooldown} seconds")
        print("🔑 Press 'q' to quit\n")
        
        frame_skip = PI_FRAME_SKIP if ENABLE_PI_OPTIMIZATIONS and is_raspberry_pi() else 1
        frame_count = 0
        
        while True:
            if CAMERA_TYPE == "picamera" and PICAMERA_AVAILABLE and is_raspberry_pi():
                frame = picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            # Run YOLO detection
            results = self.model(frame, classes=[0], verbose=False)  # class 0 is 'person'
            
            # Process detections
            person_detected = False
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Only process high-confidence detections
                    if confidence > 0.55:
                        person_detected = True
                        bbox = (x1, y1, x2, y2)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Check if we should greet this person
                        if self.should_greet(bbox):
                            self.greet_person(bbox, frame)
                            self.last_person_detected = bbox
                            self.last_description_time = time.time()
                        
                        # Only greet the first (most confident) person
                        break
                
                if person_detected:
                    break
            
            # Display the frame
            cv2.imshow('Person Detector - Office Greeter', frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        if CAMERA_TYPE == "picamera" and PICAMERA_AVAILABLE and is_raspberry_pi():
            picam2.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


def main():
    detector = PersonDetector()
    detector.run()


if __name__ == "__main__":
    main()
