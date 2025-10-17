#!/usr/bin/env python3
"""
Helper script to capture reference photos for face recognition.
Run this to easily capture photos from your webcam for each person.
"""

import cv2
import os
from pathlib import Path


def capture_photos_for_person(name, num_photos=3):
    """Capture reference photos for a person"""
    folder = f"faces/{name.lower()}"
    os.makedirs(folder, exist_ok=True)
    
    print(f"\n📸 Capturing photos for {name}")
    print(f"   Photos will be saved to: {folder}/")
    print(f"   Press SPACE to capture ({num_photos} photos needed)")
    print(f"   Press Q to skip this person\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return False
    
    captured = 0
    
    while captured < num_photos:
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
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"   ✅ Completed {name} ({captured} photos)\n")
    return True


def main():
    print("="*60)
    print("📸 Face Photo Capture Tool")
    print("="*60)
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
        capture_photos_for_person(person, num_photos=3)
    
    print("\n" + "="*60)
    print("✅ Photo capture complete!")
    print("\nNext steps:")
    print("1. Update people_config.py with the names and greetings")
    print("2. Run: uv run main.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
