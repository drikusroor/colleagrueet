"""
Configuration file for known people in the office.
Each person has attributes that help identify them using DeepFace analysis.
"""

KNOWN_PEOPLE = [
    {
        "name": "Drikus",
        "greeting": "Hello Drikus! Welcome back!",
        "features": {
            "gender": "Man",          # Man, Woman
            "age_range": (25, 40),    # age range tuple
            "race": "white",          # white, black, asian, indian, latino mediterranean, middle eastern
            "hair_color": "dark_brown",  # black, dark_brown, brown, light_brown, blonde, red
            "facial_hair": True,      # True if has beard/mustache
            "glasses": True,          # True if wears glasses
        }
    },
    {
        "name": "Robert",
        "greeting": "Hey Robert! Good to see you!",
        "features": {
            "gender": "Man",
            "age_range": (25, 40),
            "race": "white",
            "hair_color": "blonde",   # big blonde/orangey beard
            "facial_hair": True,
            "glasses": False,
        }
    },
    {
        "name": "Mohamed",
        "greeting": "Welcome Mohamed!",
        "features": {
            "gender": "Man",
            "age_range": (25, 35),
            "race": "middle eastern",         # brown skin
            "hair_color": "black",
            "facial_hair": True,
            "glasses": False,
        }
    },
    {
        "name": "Adriana",
        "greeting": "Hi Adriana! Nice to see you!",
        "features": {
            "gender": "Woman",
            "age_range": (25, 35),
            "race": "latino mediterranean",
            "hair_color": "brown",
            "facial_hair": False,
            "glasses": False,
        }
    },
]

# Matching thresholds
MATCH_THRESHOLD = 3.5  # Minimum score that must be achieved (out of 8 possible points)
UNKNOWN_GREETING = "Hello! I don't recognize you. Please check in at reception."

# Age tolerance for matching
AGE_TOLERANCE = 8  # years +/- for age matching

