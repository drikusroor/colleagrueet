# Person Detector

A Python application that uses YOLO (You Only Look Once) to detect people in real-time via webcam and describes them using text-to-speech.

## Features

- Real-time person detection using YOLOv8
- Text-to-speech descriptions of detected people
- Describes position (left/center/right) and distance from camera
- Visual bounding boxes with confidence scores
- Cooldown period to avoid repetitive announcements

## Requirements

- Python 3.9+
- Webcam
- uv package manager

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management:

```bash
# Dependencies are already configured in pyproject.toml
# Just sync the environment
uv sync
```

## Usage

Run the application:

```bash
uv run main.py
```

The application will:
1. Open your webcam
2. Display a window showing the video feed
3. Draw green bounding boxes around detected people
4. Announce descriptions via text-to-speech when a person is detected
5. Continue until you press 'q' to quit

## How It Works

- Uses YOLOv8 nano model for fast, efficient person detection
- Analyzes bounding box position and size to generate descriptions
- Implements a 10-second cooldown between announcements for the same person
- Processes only high-confidence detections (>50% confidence)

## Controls

- **q**: Quit the application
