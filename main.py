import cv2
import pyttsx3
from ultralytics import YOLO
import time
import numpy as np
import threading


class PersonDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model for speed
        self.tts_engine = pyttsx3.init()
        self.last_person_detected = None
        self.last_description_time = 0
        self.description_cooldown = 15  # seconds between descriptions
        self.is_speaking = False
        self.lock = threading.Lock()
        
    def analyze_appearance(self, person_crop):
        """Analyze person's appearance from the cropped image"""
        # Get average color of upper body (approximate clothing color)
        h, w = person_crop.shape[:2]
        upper_body = person_crop[int(h*0.2):int(h*0.5), :]
        
        # Analyze hair region (top portion)
        hair_region = person_crop[0:int(h*0.2), :]
        
        # Get dominant colors
        avg_color = np.mean(upper_body, axis=(0, 1))
        hair_color = np.mean(hair_region, axis=(0, 1))
        
        # Determine clothing color description
        b, g, r = avg_color
        if r > 150 and g < 100 and b < 100:
            clothing = "wearing red"
        elif r < 100 and g < 100 and b > 150:
            clothing = "wearing blue"
        elif r < 100 and g > 150 and b < 100:
            clothing = "wearing green"
        elif r > 200 and g > 200 and b > 200:
            clothing = "wearing white or light colored clothing"
        elif r < 80 and g < 80 and b < 80:
            clothing = "wearing dark clothing"
        else:
            clothing = "wearing casual clothing"
        
        # Determine hair color
        hr, hg, hb = hair_color
        if hr < 60 and hg < 60 and hb < 60:
            hair = "with dark hair"
        elif hr > 150 and hg > 120 and hb > 80:
            hair = "with light hair"
        elif hr > 100 and hg < 80 and hb < 80:
            hair = "with reddish hair"
        else:
            hair = "with brown hair"
        
        return clothing, hair
        
    def describe_person(self, bbox, frame):
        """Generate and speak a description of the detected person"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Extract person crop for analysis
        person_crop = frame[y1:y2, x1:x2]
        
        # Determine position
        frame_width = frame.shape[1]
        center_x = (x1 + x2) / 2
        if center_x < frame_width / 3:
            position = "on the left"
        elif center_x < 2 * frame_width / 3:
            position = "in the center"
        else:
            position = "on the right"
        
        # Determine distance
        if height > 400:
            distance = "close to the camera"
        elif height > 200:
            distance = "at medium distance"
        else:
            distance = "far away"
        
        # Analyze appearance
        clothing, hair = self.analyze_appearance(person_crop)
        
        description = f"I see a person {position}, {distance}, {clothing} {hair}"
        print(f"Speaking: {description}")
        
        # Speak in a separate thread to not block detection
        with self.lock:
            if not self.is_speaking:
                self.is_speaking = True
                threading.Thread(target=self._speak, args=(description,), daemon=True).start()
    
    def _speak(self, text):
        """Speak text in a separate thread"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        finally:
            with self.lock:
                self.is_speaking = False
    
    def should_describe(self, bbox):
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
            
            # Only describe if moved significantly (>100 pixels)
            if movement < 100:
                return False
        
        return True
        
    def run(self):
        """Main loop for webcam detection"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting person detection. Press 'q' to quit.")
        print(f"Descriptions will be given every {self.description_cooldown} seconds.")
        
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
                        
                        # Check if we should describe this person
                        if self.should_describe(bbox):
                            self.describe_person(bbox, frame)
                            self.last_person_detected = bbox
                            self.last_description_time = time.time()
                        
                        # Only describe the first (most confident) person
                        break
                
                if person_detected:
                    break
            
            # Display the frame
            cv2.imshow('Person Detector', frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


def main():
    detector = PersonDetector()
    detector.run()



if __name__ == "__main__":
    main()
