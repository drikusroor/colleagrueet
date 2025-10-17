# Raspberry Pi Setup Guide

This guide will help you set up the face recognition system on a Raspberry Pi 4 with the Camera Module.

## Hardware Requirements

- Raspberry Pi 4 (2GB+ RAM recommended, 4GB+ ideal)
- Raspberry Pi Camera Module (v1, v2, or v3)
- MicroSD card (32GB+ recommended)
- Power supply for Raspberry Pi
- (Optional) Heatsink or fan for cooling during intensive operations

## Software Setup

### 1. Install Raspberry Pi OS

1. Use Raspberry Pi Imager to install **Raspberry Pi OS (64-bit)** on your SD card
2. Enable SSH and configure WiFi during imaging if needed
3. Boot up your Raspberry Pi

### 2. Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### 3. Enable Camera

```bash
# Enable the camera interface
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Reboot
sudo reboot
```

### 4. Install System Dependencies

```bash
# Install picamera2 and related packages (REQUIRED for Pi Camera)
sudo apt install -y python3-picamera2 python3-opencv

# Install other system dependencies
sudo apt install -y python3-pip python3-venv
sudo apt install -y libatlas-base-dev libhdf5-dev
sudo apt install -y portaudio19-dev python3-pyaudio

# For audio output (gTTS)
sudo apt install -y mpg123
```

**Important**: `picamera2` must be installed via system packages on Raspberry Pi because it has Linux-specific dependencies that won't install via pip/uv on macOS/Windows.

### 5. Install UV (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 6. Clone/Copy Your Project

```bash
# If using git:
git clone <your-repo-url> describe-drikus
cd describe-drikus

# Or copy files manually via scp/rsync
```

### 7. Install Python Dependencies

```bash
# This will automatically install picamera2 on ARM64 systems
uv sync
```

## Configuration

### 1. Camera Settings

Edit `config.py`:

```python
# For Raspberry Pi Camera Module
CAMERA_TYPE = "picamera"  # or "auto" to auto-detect

# Adjust resolution for better performance
PICAMERA_RESOLUTION = (640, 480)  # Lower for better performance
PICAMERA_FRAMERATE = 15

# Enable Pi optimizations
ENABLE_PI_OPTIMIZATIONS = True
PI_FRAME_SKIP = 2  # Process every 2nd frame (adjust based on performance)
```

### 2. Performance Tuning

For Raspberry Pi 4, you may want to adjust these settings:

```python
# In config.py
PERSON_DETECTION_CONFIDENCE = 0.7  # Slightly higher to reduce false positives
PI_FRAME_SKIP = 3  # Process every 3rd frame if performance is slow
```

## Running the Application

### 1. Capture Face Photos

```bash
# Run with Pi Camera
uv run capture_faces.py
```

The script will auto-detect the Raspberry Pi and ask if you want to use the Pi Camera.

### 2. Run Face Recognition

```bash
# Run the main application
uv run main.py
```

### 3. Run on Boot (Optional)

Create a systemd service:

```bash
sudo nano /etc/systemd/system/face-greeter.service
```

Add:

```ini
[Unit]
Description=Face Recognition Greeter
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/describe-drikus
ExecStart=/home/pi/.cargo/bin/uv run main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable face-greeter.service
sudo systemctl start face-greeter.service

# Check status
sudo systemctl status face-greeter.service

# View logs
sudo journalctl -u face-greeter.service -f
```

## Troubleshooting

### Camera Not Detected

```bash
# Test camera
libcamera-hello

# If that works but Python doesn't see it:
python3 -c "from picamera2 import Picamera2; print(Picamera2())"
```

### Performance Issues

1. **Reduce resolution**: Set `PICAMERA_RESOLUTION = (320, 240)` in config.py
2. **Increase frame skip**: Set `PI_FRAME_SKIP = 4` or higher
3. **Add cooling**: Use a heatsink or fan on the Pi
4. **Overclock (careful)**: Use `sudo raspi-config` to overclock (voids warranty)

### Memory Issues

```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Audio Not Working

```bash
# Test audio output
speaker-test -c2

# Configure audio output (headphone jack vs HDMI)
sudo raspi-config
# System Options → Audio
```

### Display Issues (Headless Mode)

If running without a display, you can disable OpenCV windows:

Edit `main.py` and comment out:
```python
# cv2.imshow('Person Detector - Office Greeter', frame)
```

## Performance Expectations

On a Raspberry Pi 4 (4GB):
- **First run**: 2-5 minutes to generate face embeddings (one-time)
- **Detection speed**: 1-3 FPS with frame skipping
- **Recognition time**: 1-2 seconds per person

To improve performance:
- Use lower resolution (320x240 or 640x480)
- Increase frame skip (3-5 frames)
- Ensure good cooling
- Close other applications

## Camera Module Tips

1. **Positioning**: Mount at eye level, 3-6 feet from where people will stand
2. **Lighting**: Ensure even, bright lighting (avoid backlighting)
3. **Focus**: Camera Module v1/v2 have fixed focus; v3 has autofocus
4. **Angle**: Point slightly downward for best face capture

## Security Notes

- Change default passwords
- Use strong WiFi passwords
- Consider disabling SSH when not needed
- Keep the system updated: `sudo apt update && sudo apt upgrade`

## Additional Resources

- [Raspberry Pi Camera Documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [Picamera2 Library](https://github.com/raspberrypi/picamera2)
- [Raspberry Pi Performance Tuning](https://www.raspberrypi.com/documentation/computers/config_txt.html)
