# Office Person Greeter

A Python application that uses YOLO (You Only Look Once) to detect and identify people entering your office, greeting them by name using natural text-to-speech.

## Features

- Real-time person detection using YOLOv8
- Advanced feature detection (skin tone, hair color/length, facial hair, build)
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

## Configuration

Edit `people_config.py` to add your office mates:

```python
KNOWN_PEOPLE = [
    {
        "name": "John",
        "greeting": "Hello John! Welcome back!",
        "features": {
            "skin_tone": "light",      # light, medium, dark
            "hair_color": "dark",       # dark, light, brown, red
            "hair_length": "short",     # short, medium, long
            "facial_hair": "beard",     # none, mustache, beard
            "glasses": True,
            "build": "average"          # slim, average, heavy
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
1. Open your webcam
2. Display a window showing the video feed
3. Detect people entering the office
4. Analyze their features (hair, facial hair, build, skin tone)
5. Match them against known people
6. Greet them by name with a personalized message
7. Alert if an unknown person is detected
8. Continue until you press 'q' to quit

## How It Works

- **Detection**: Uses YOLOv8 nano model for fast, efficient person detection
- **Feature Analysis**: Analyzes detected person's appearance:
  - Skin tone (light/medium/dark)
  - Hair color (dark/light/brown/red)
  - Hair length (short/medium/long)
  - Facial hair (none/mustache/beard)
  - Build (slim/average/heavy)
- **Matching**: Scores detected features against known people profiles
- **Greeting**: Uses Google TTS for natural-sounding greetings
- **Security**: Identifies unknown people with a different message
- **Cooldown**: 15-second interval and movement detection prevents spam

## Controls

- **q**: Quit the application

## Security Features

When an unknown person is detected, the system:
- Plays a different greeting asking them to check in at reception
- Logs the detection with feature details
- Could be extended to send notifications/alerts
