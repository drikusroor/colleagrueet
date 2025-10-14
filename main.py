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
from people_config import KNOWN_PEOPLE, MATCH_THRESHOLD, UNKNOWN_GREETING, AGE_TOLERANCE


class PersonDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        pygame.mixer.init()
        self.last_person_detected = None
        self.last_description_time = 0
        self.description_cooldown = 15  # seconds between greetings
        self.is_speaking = False
        self.lock = threading.Lock()
        self.greeted_people = set()  # Track who we've greeted recently
        print("Warming up DeepFace models...")
        # Warm up DeepFace to download models on first run
        try:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(dummy, actions=['age', 'gender', 'race', 'emotion'], 
                           enforce_detection=False, silent=True)
        except:
            pass
        print("DeepFace ready!")
        
    def analyze_features(self, person_crop):
        """Analyze person's features using DeepFace"""
        try:
            # Resize image for better detection
            h, w = person_crop.shape[:2]
            if h < 100 or w < 100:
                return None
            
            # First, try to detect face to get better analysis
            try:
                face_objs = DeepFace.extract_faces(person_crop, 
                                                   detector_backend='opencv',
                                                   enforce_detection=False,
                                                   align=True)
                if face_objs and len(face_objs) > 0:
                    # Get the face with highest confidence
                    face_obj = max(face_objs, key=lambda x: x.get('confidence', 0))
                    facial_area = face_obj['facial_area']
                    x, y, w_face, h_face = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Extract face with some padding for context
                    padding = 30
                    y1 = max(0, y - padding)
                    y2 = min(h, y + h_face + padding)
                    x1 = max(0, x - padding)
                    x2 = min(w, x + w_face + padding)
                    face_crop = person_crop[y1:y2, x1:x2]
                    
                    # Use face crop for analysis
                    analysis = DeepFace.analyze(face_crop, 
                                               actions=['age', 'gender', 'race'],
                                               enforce_detection=False,
                                               detector_backend='skip',  # Already detected
                                               silent=True)
                else:
                    # No face detected, use full crop
                    analysis = DeepFace.analyze(person_crop, 
                                               actions=['age', 'gender', 'race'],
                                               enforce_detection=False,
                                               silent=True)
            except:
                # Fallback to full crop analysis
                analysis = DeepFace.analyze(person_crop, 
                                           actions=['age', 'gender', 'race'],
                                           enforce_detection=False,
                                           silent=True)
            
            # DeepFace returns a list, get first result
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            features = {}
            
            # Extract gender (DeepFace returns "Man" or "Woman")
            features["gender"] = analysis.get("dominant_gender", "Unknown")
            
            # Extract age
            features["age"] = int(analysis.get("age", 0))
            
            # Extract race (DeepFace returns: asian, indian, black, white, middle eastern, latino mediterranean)
            race = analysis.get("dominant_race", "unknown")
            features["race"] = race.lower()
            
            # Analyze hair color from top portion of person
            hair_region = person_crop[0:int(h*0.15), :]
            if hair_region.size > 0:
                hair_avg = np.mean(hair_region, axis=(0, 1))
                b_hair, g_hair, r_hair = hair_avg
                
                # Determine hair color
                if r_hair < 50 and g_hair < 50 and b_hair < 50:
                    features["hair_color"] = "black"
                elif r_hair > 140 and g_hair > 120 and b_hair > 80:
                    features["hair_color"] = "blonde"
                elif r_hair > 120 and g_hair < 90 and b_hair < 70:
                    features["hair_color"] = "red"
                elif r_hair > 100 and g_hair > 80 and b_hair < 80:
                    features["hair_color"] = "brown"
                elif r_hair > 100 and g_hair > 100 and b_hair > 100:
                    features["hair_color"] = "light_brown"
                else:
                    features["hair_color"] = "dark_brown"
            else:
                features["hair_color"] = "unknown"
            
            # Detect facial hair by checking if there's significant dark area in lower face
            lower_face_region = person_crop[int(h*0.4):int(h*0.6), :]
            if lower_face_region.size > 0:
                gray_lower = cv2.cvtColor(lower_face_region, cv2.COLOR_BGR2GRAY)
                dark_pixels = np.sum(gray_lower < 70)
                total_pixels = gray_lower.size
                features["facial_hair"] = (dark_pixels / total_pixels) > 0.25
            else:
                features["facial_hair"] = False
            
            # Better glasses detection - look for rectangular frame patterns and reflections
            eye_region = person_crop[int(h*0.25):int(h*0.45), :]
            if eye_region.size > 0:
                # Convert to grayscale for edge detection
                gray_eyes = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                
                # Check for bright reflections (typical of glasses)
                bright_pixels = np.sum(gray_eyes > 210)
                
                # Check for edges (frame detection)
                edges = cv2.Canny(gray_eyes, 50, 150)
                edge_pixels = np.sum(edges > 0)
                
                total_pixels = gray_eyes.size
                
                # Glasses likely if significant bright spots OR many edges
                has_reflections = (bright_pixels / total_pixels) > 0.03
                has_frames = (edge_pixels / total_pixels) > 0.15
                
                features["glasses"] = has_reflections or has_frames
            else:
                features["glasses"] = False
            
            return features
            
        except Exception as e:
            print(f"DeepFace analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def match_person(self, detected_features):
        """Match detected features against known people"""
        if detected_features is None:
            return None, 0
            
        best_match = None
        best_score = 0
        match_details = []
        
        for person in KNOWN_PEOPLE:
            score = 0
            details = []
            person_features = person["features"]
            
            # Gender match (CRITICAL - must match!) - 3 points
            detected_gender = detected_features.get("gender", "").lower()
            expected_gender = person_features.get("gender", "").lower()
            if detected_gender == expected_gender:
                score += 3
                details.append(f"✓ gender ({detected_gender})")
            else:
                # Gender mismatch is a deal-breaker - skip this person
                details.append(f"✗ gender ({detected_gender} ≠ {expected_gender})")
                match_details.append((person["name"], 0, details))
                continue
            
            # Race match (very important) - 2 points
            detected_race = detected_features.get("race", "").lower()
            expected_race = person_features.get("race", "").lower()
            if detected_race == expected_race:
                score += 2
                details.append(f"✓ race ({detected_race})")
            else:
                details.append(f"✗ race ({detected_race} ≠ {expected_race})")
            
            # Hair color match - 1 point
            detected_hair = detected_features.get("hair_color", "").lower()
            expected_hair = person_features.get("hair_color", "").lower()
            if detected_hair == expected_hair:
                score += 1
                details.append(f"✓ hair_color ({detected_hair})")
            else:
                details.append(f"✗ hair_color ({detected_hair} ≠ {expected_hair})")
            
            # Age match (with tolerance) - 1 point
            detected_age = detected_features.get("age", 0)
            age_min, age_max = person_features.get("age_range", (0, 100))
            if age_min - AGE_TOLERANCE <= detected_age <= age_max + AGE_TOLERANCE:
                score += 1
                details.append(f"✓ age ({detected_age} in {age_min}-{age_max})")
            else:
                details.append(f"✗ age ({detected_age} not in {age_min}-{age_max})")
            
            # Facial hair match - 0.5 points
            if detected_features.get("facial_hair") == person_features.get("facial_hair"):
                score += 0.5
                details.append(f"✓ facial_hair")
            else:
                details.append(f"✗ facial_hair")
            
            # Glasses match - 0.5 points
            if detected_features.get("glasses") == person_features.get("glasses"):
                score += 0.5
                details.append(f"✓ glasses")
            else:
                details.append(f"✗ glasses")
            
            match_details.append((person["name"], score, details))
            
            if score > best_score:
                best_score = score
                best_match = person
        
        # Print detailed matching info
        print("\nMatching results:")
        for name, score, details in sorted(match_details, key=lambda x: x[1], reverse=True):
            print(f"  {name}: {score:.1f}/8.0 - {', '.join(details)}")
        
        # Return match only if score is above threshold
        if best_score >= MATCH_THRESHOLD:
            return best_match, best_score
        else:
            return None, best_score
        
    def greet_person(self, bbox, frame):
        """Identify and greet the person"""
        x1, y1, x2, y2 = bbox
        
        # Extract person crop for analysis
        person_crop = frame[y1:y2, x1:x2]
        
        # Analyze features using DeepFace
        print("\n" + "="*60)
        print("🔍 Analyzing person with DeepFace...")
        detected_features = self.analyze_features(person_crop)
        
        if detected_features is None:
            print("❌ Could not analyze person features")
            return
        
        print(f"\n📊 Detected Features:")
        print(f"  Gender: {detected_features['gender']}")
        print(f"  Age: {detected_features['age']} years")
        print(f"  Race: {detected_features['race']}")
        print(f"  Hair color: {detected_features['hair_color']}")
        print(f"  Facial hair: {'Yes' if detected_features['facial_hair'] else 'No'}")
        print(f"  Glasses: {'Yes' if detected_features['glasses'] else 'No'}")
        
        # Match against known people
        matched_person, score = self.match_person(detected_features)
        
        if matched_person:
            greeting = matched_person["greeting"]
            person_name = matched_person["name"]
            print(f"\n✅ MATCHED: {person_name} (score: {score:.1f}/8.0)")
        else:
            greeting = UNKNOWN_GREETING
            person_name = "Unknown"
            print(f"\n⚠️  UNKNOWN PERSON (best score: {score:.1f}/8.0 - below threshold {MATCH_THRESHOLD})")
        
        print(f"🗣️  Speaking: {greeting}\n")
        
        # Speak the greeting
        with self.lock:
            if not self.is_speaking:
                self.is_speaking = True
                threading.Thread(target=self._speak, args=(greeting,), daemon=True).start()
                self.greeted_people.add(person_name)
    
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
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting person detection and greeting system.")
        print(f"Known people: {len(KNOWN_PEOPLE)}")
        for person in KNOWN_PEOPLE:
            features = person['features']
            print(f"  - {person['name']}: {features['gender']}, age {features['age_range'][0]}-{features['age_range'][1]}, "
                  f"{features['race']}, {'beard' if features['facial_hair'] else 'no beard'}, "
                  f"{'glasses' if features['glasses'] else 'no glasses'}")
        print(f"\nGreetings will be given every {self.description_cooldown} seconds.")
        print("Using DeepFace for accurate facial analysis (gender, age, race)")
        print("Press 'q' to quit.\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
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
                    if confidence > 0.6:
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
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


def main():
    detector = PersonDetector()
    detector.run()



if __name__ == "__main__":
    main()
