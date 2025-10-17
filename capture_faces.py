#!/usr/bin/env python3
"""
Helper script to capture reference photos for face recognition.
Run this to easily capture photos from your webcam for each person.
"""

import cv2
import os
from pathlib import Path
import sys

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


def capture_photos_for_person(name, num_photos=3, use_picamera=False):
    """Capture reference photos for a person"""
    folder = f"faces/{name.lower()}"
    os.makedirs(folder, exist_ok=True)
    
    print(f"\n📸 Capturing photos for {name}")
    print(f"   Photos will be saved to: {folder}/")
    print(f"   Press SPACE to capture ({num_photos} photos needed)")
    print(f"   Press Q to skip this person\n")
    
    if use_picamera and PICAMERA_AVAILABLE:
        print("   Using Raspberry Pi Camera...")
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}))
        picam2.start()
        cap = None
    else:
        print("   Using USB webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return False
        picam2 = None
    
    captured = 0
    
    while captured < num_photos:
        if use_picamera and picam2:
            frame = picam2.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
        
        # Display instructions
        cv2.putText(frame, f"Capturing for: {name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Photo {captured + 1}/{num_photos}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture, Q to skip", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture Photos', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar
            filename = f"{folder}/photo{captured + 1}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   ✓ Saved: {filename}")
            captured += 1
        elif key == ord('q'):  # Q to skip
            print(f"   ⏭  Skipped {name}")
            if picam2:
                picam2.stop()
            elif cap:
                cap.release()
            cv2.destroyAllWindows()
            return False
    
    if picam2:
        picam2.stop()
    elif cap:
        cap.release()
    cv2.destroyAllWindows()
    
    print(f"   ✅ Completed {name} ({captured} photos)\n")
    return True


def main():
    print("="*60)
    print("📸 Face Photo Capture Tool")
    print("="*60)
    
    # Detect camera type
    use_picamera = False
    if is_raspberry_pi() and PICAMERA_AVAILABLE:
        print("\n🍓 Raspberry Pi detected!")
        response = input("Use Raspberry Pi Camera? (y/n, default: y): ").strip().lower()
        use_picamera = response != 'n'
    elif is_raspberry_pi() and not PICAMERA_AVAILABLE:
        print("\n🍓 Raspberry Pi detected, but picamera2 not installed.")
        print("   Install with: sudo apt install -y python3-picamera2")
        print("   Falling back to USB webcam...")
    
    print("\nThis tool will help you capture reference photos for each person.")
    print("Tips:")
    print("  • Look directly at the camera")
    print("  • Ensure good lighting")
    print("  • Try different angles/expressions for better accuracy")
    print("  • Remove glasses if you don't usually wear them")
    print("\n" + "="*60)
    
    people = []
    while True:
        name = input("\nEnter person's name (or press Enter to finish): ").strip()
        if not name:
            break
        people.append(name)
    
    if not people:
        print("\n❌ No people added. Exiting.")
        return
    
    print(f"\n✓ Will capture photos for {len(people)} people: {', '.join(people)}")
    
    for person in people:
        capture_photos_for_person(person, num_photos=3, use_picamera=use_picamera)
    
    print("\n" + "="*60)
    print("✅ Photo capture complete!")
    print("\nNext steps:")
    print("1. Update people_config.py with the names and greetings")
    print("2. Run: uv run main.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
