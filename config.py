"""
Global configuration for the face recognition system.
Person-specific settings are in faces/{name}/config.txt
"""

# Camera settings
CAMERA_TYPE = "auto"  # Options: "auto", "usb", "picamera"
PICAMERA_RESOLUTION = (640, 480)  # Resolution for Pi Camera (lower for better performance)
PICAMERA_FRAMERATE = 15  # Framerate for Pi Camera (lower for better performance)

# Recognition settings
DEFAULT_SIMILARITY_THRESHOLD = 0.40  # Lower = more strict (typical: 0.3-0.5 for cosine distance)
UNKNOWN_GREETING = "Hello! I don't recognize you. Please check in at reception."
FACE_MODEL = "Facenet512"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace

# Detection settings
PERSON_DETECTION_CONFIDENCE = 0.6  # YOLO confidence threshold
GREETING_COOLDOWN_SECONDS = 15  # Time between greetings
MOVEMENT_THRESHOLD_PIXELS = 150  # Movement required to re-greet

# Raspberry Pi optimizations
ENABLE_PI_OPTIMIZATIONS = True  # Enable when running on Raspberry Pi
PI_FRAME_SKIP = 2  # Process every Nth frame on Pi (higher = better performance, lower accuracy)
