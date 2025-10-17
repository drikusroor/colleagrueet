"""
Global configuration for the face recognition system.
Person-specific settings are in faces/{name}/config.txt
"""

# Recognition settings
DEFAULT_SIMILARITY_THRESHOLD = 0.40  # Lower = more strict (typical: 0.3-0.5 for cosine distance)
UNKNOWN_GREETING = "Hello! I don't recognize you. Please check in at reception."
FACE_MODEL = "Facenet512"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace

# Detection settings
PERSON_DETECTION_CONFIDENCE = 0.6  # YOLO confidence threshold
GREETING_COOLDOWN_SECONDS = 15  # Time between greetings
MOVEMENT_THRESHOLD_PIXELS = 150  # Movement required to re-greet
