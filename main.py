import cv2
from gtts import gTTS
import pygame
from ultralytics import YOLO
import time
import numpy as np
import threading
import os
import tempfile
from people_config import KNOWN_PEOPLE, MATCH_THRESHOLD, UNKNOWN_GREETING


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
        
    def analyze_features(self, person_crop, face_region):
        """Analyze person's features for identification"""
        h, w = person_crop.shape[:2]
        
        # Analyze face/head region for skin tone and hair
        face_crop = person_crop[0:int(h*0.3), :]
        
        # Get average colors from different regions
        face_avg = np.mean(face_crop, axis=(0, 1))
        
        # Hair region (top 15% of person)
        hair_region = person_crop[0:int(h*0.15), :]
        hair_avg = np.mean(hair_region, axis=(0, 1))
        
        # Body region for build estimation (middle section)
        body_region = person_crop[int(h*0.3):int(h*0.7), :]
        
        features = {}
        
        # Detect skin tone
        b_face, g_face, r_face = face_avg
        brightness = (r_face + g_face + b_face) / 3
        if brightness > 180:
            features["skin_tone"] = "light"
        elif brightness > 120:
            features["skin_tone"] = "medium"
        else:
            features["skin_tone"] = "dark"
        
        # Detect hair color
        b_hair, g_hair, r_hair = hair_avg
        if r_hair < 60 and g_hair < 60 and b_hair < 60:
            features["hair_color"] = "dark"
        elif r_hair > 150 and g_hair > 130:
            features["hair_color"] = "light"  # blonde/light
        elif r_hair > 120 and g_hair < 100:
            features["hair_color"] = "red"
        else:
            features["hair_color"] = "brown"
        
        # Estimate hair length (based on how far down hair color extends)
        hair_extend = int(h * 0.25)
        lower_hair = person_crop[int(h*0.15):hair_extend, :]
        lower_hair_avg = np.mean(lower_hair, axis=(0, 1))
        
        # If similar color extends down, likely longer hair
        color_diff = np.abs(hair_avg - lower_hair_avg).mean()
        if color_diff < 30:
            features["hair_length"] = "long"
        elif h > 300:
            features["hair_length"] = "medium"
        else:
            features["hair_length"] = "short"
        
        # Detect facial hair (darker region in lower face)
        lower_face = person_crop[int(h*0.15):int(h*0.3), :]
        lower_face_avg = np.mean(lower_face, axis=(0, 1))
        lower_brightness = lower_face_avg.mean()
        
        if lower_brightness < brightness - 30:
            features["facial_hair"] = "beard"
        elif lower_brightness < brightness - 15:
            features["facial_hair"] = "mustache"
        else:
            features["facial_hair"] = "none"
        
        # Estimate build based on width-to-height ratio
        aspect_ratio = w / h
        if aspect_ratio > 0.55:
            features["build"] = "heavy"
        elif aspect_ratio > 0.4:
            features["build"] = "average"
        else:
            features["build"] = "slim"
        
        # Glasses detection (simplified - look for bright reflections in face region)
        # This is a basic heuristic
        features["glasses"] = False  # Default, would need better detection
        
        return features
    
    def match_person(self, detected_features):
        """Match detected features against known people"""
        best_match = None
        best_score = 0
        
        for person in KNOWN_PEOPLE:
            score = 0
            person_features = person["features"]
            
            # Score each matching feature
            if detected_features.get("skin_tone") == person_features.get("skin_tone"):
                score += 2  # Skin tone is important
            if detected_features.get("hair_color") == person_features.get("hair_color"):
                score += 2  # Hair color is important
            if detected_features.get("hair_length") == person_features.get("hair_length"):
                score += 1
            if detected_features.get("facial_hair") == person_features.get("facial_hair"):
                score += 2  # Facial hair is distinctive
            if detected_features.get("build") == person_features.get("build"):
                score += 1
            if detected_features.get("glasses") == person_features.get("glasses"):
                score += 1
            
            if score > best_score:
                best_score = score
                best_match = person
        
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
        
        # Analyze features
        detected_features = self.analyze_features(person_crop, None)
        
        # Match against known people
        matched_person, score = self.match_person(detected_features)
        
        if matched_person:
            greeting = matched_person["greeting"]
            person_name = matched_person["name"]
            print(f"Recognized: {person_name} (confidence: {score})")
            print(f"Detected features: {detected_features}")
        else:
            greeting = UNKNOWN_GREETING
            person_name = "Unknown"
            print(f"Unknown person detected (score: {score})")
            print(f"Detected features: {detected_features}")
        
        print(f"Speaking: {greeting}")
        
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
            print(f"  - {person['name']}")
        print(f"\nGreetings will be given every {self.description_cooldown} seconds.")
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
