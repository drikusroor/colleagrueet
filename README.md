# Office Person Greeter

A Python application that uses YOLO for person detection and DeepFace for facial analysis to identify and greet people entering your office by name using natural text-to-speech.

## Features

- Real-time person detection using YOLOv8
- **Professional facial analysis using DeepFace**:
  - Gender detection (Man/Woman)
  - Age estimation
  - Race/ethnicity detection
  - Facial hair detection (heuristic)
  - Glasses detection (heuristic)
- Person identification matching against configured profiles
- Natural text-to-speech greetings using Google TTS
- Unknown person detection for security alerts
- Visual bounding boxes with confidence scores
- Smart cooldown to avoid repetitive greetings

## Requirements

- Python 3.9+
- Webcam
- Internet connection (for Google TTS)
- uv package manager

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management:

```bash
# Dependencies are already configured in pyproject.toml
# Just sync the environment
uv sync
```

**Note**: On first run, DeepFace will download AI models (~100MB) for facial analysis. This is a one-time setup.

## Configuration

Edit `people_config.py` to add your office mates:

```python
KNOWN_PEOPLE = [
    {
        "name": "Drikus",
        "greeting": "Hello Drikus! Welcome back!",
        "features": {
            "gender": "Man",              # Man, Woman
            "age_range": (25, 35),        # age range tuple
            "race": "white",              # white, black, asian, indian, 
                                          # latino mediterranean, middle eastern
            "facial_hair": True,          # True if has beard/mustache
            "glasses": True,              # True if wears glasses
        }
    }
]
```

## Usage

Run the application:

```bash
uv run main.py
```

The application will:
1. Warm up DeepFace models (first run downloads models)
2. Open your webcam
3. Display a window showing the video feed
4. Detect people entering the office
5. Analyze their features using AI (gender, age, race, facial hair, glasses)
6. Match them against known people with scoring system
7. Greet them by name with a personalized message
8. Alert if an unknown person is detected
9. Continue until you press 'q' to quit

## How It Works

- **Detection**: Uses YOLOv8 nano model for fast, efficient person detection
- **Facial Analysis**: Uses DeepFace (state-of-the-art facial recognition framework) to detect:
  - Gender (Man/Woman)
  - Age (estimated years)
  - Race/Ethnicity (white, black, asian, indian, latino mediterranean, middle eastern)
  - Facial hair (beard detection via image analysis)
  - Glasses (bright reflection detection)
- **Matching**: Scores detected features against known people profiles:
  - Gender match: 2 points
  - Race match: 2 points
  - Age match (±8 years tolerance): 1 point
  - Facial hair match: 0.5 points
  - Glasses match: 0.5 points
  - Threshold: 2.5 points minimum to identify
- **Greeting**: Uses Google TTS for natural-sounding greetings
- **Security**: Identifies unknown people with a different message
- **Cooldown**: 15-second interval and movement detection prevents spam

## Controls

- **q**: Quit the application

## Security Features

When an unknown person is detected, the system:
- Plays a different greeting asking them to check in at reception
- Logs the detection with detailed feature analysis
- Could be extended to send notifications/alerts

## Technical Details

- **YOLO**: Fast object detection for real-time person tracking
- **DeepFace**: State-of-the-art facial attribute analysis
- **Google TTS**: Natural-sounding voice synthesis
- **Pygame**: Audio playback
- **Threading**: Non-blocking speech for continuous detection
