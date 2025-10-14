"""
Configuration file for known people in the office.
Each person has attributes that help identify them.
"""

KNOWN_PEOPLE = [
    {
        "name": "Drikus",
        "greeting": "Hello Drikus! Welcome back!",
        "features": {
            "skin_tone": "light",  # light, medium, dark
            "hair_color": "dark",  # dark, light, brown, red, gray
            "hair_length": "short",  # short, medium, long
            "facial_hair": "beard",  # none, mustache, beard, goatee
            "glasses": True,
            "build": "average"  # slim, average, heavy
        }
    },
    {
        "name": "Robert",
        "greeting": "Hey Robert! Good to see you!",
        "features": {
            "skin_tone": "light",
            "hair_color": "light",  # blonde
            "hair_length": "long",
            "facial_hair": "beard",  # big blonde/orangey beard
            "glasses": False,
            "build": "heavy"
        }
    },
    {
        "name": "Mohamed",
        "greeting": "Welcome Mohamed!",
        "features": {
            "skin_tone": "dark",
            "hair_color": "dark",
            "hair_length": "short",
            "facial_hair": "none",
            "glasses": False,
            "build": "average"
        }
    }
]

# Matching thresholds
MATCH_THRESHOLD = 3  # Minimum features that must match
UNKNOWN_GREETING = "Hello! I don't recognize you. Please check in at reception."
