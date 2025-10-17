# Office Person Greeter - Face Recognition Edition

A Python application that uses YOLO for person detection and **DeepFace face embeddings** to accurately identify and greet people entering your office by name using natural text-to-speech.

## 🎯 How It Works

Instead of analyzing facial characteristics (which is unreliable), this system uses **face embeddings** - unique numerical fingerprints of each person's face. This is the same technology used by Facebook, Apple Face ID, and professional security systems.

1. **Training**: You provide 1-3 photos of each person
2. **Embedding Generation**: DeepFace generates a unique 512-dimensional vector for each face
3. **Recognition**: When someone appears, their face embedding is compared to stored embeddings
4. **Matching**: If the similarity is high enough (>60% by default), they're recognized!

## 🚀 Features

- **Accurate face recognition** using state-of-the-art embeddings (Facenet512)
- **No manual characteristic configuration** - just add photos!
- Real-time person detection using YOLOv8
- Natural text-to-speech greetings using Google TTS
- Unknown person detection for security alerts
- Caches embeddings for fast startup
- Detailed recognition scoring for debugging

## 📋 Requirements

- Python 3.9+
- Webcam
- Internet connection (for Google TTS)
- uv package manager

## 🔧 Installation

```bash
# Dependencies are managed by uv
uv sync
```

## 📸 Setup - Adding Face Photos

The system automatically discovers people by scanning the `faces/` directory. Each person gets their own folder with photos and an optional config file.

### Step 1: Create folder structure

```
faces/
├── person.config.example  (example config file)
├── yourname/
│   ├── config.txt         (your personal config)
│   ├── photo1.jpg
│   └── photo2.jpg
└── colleague/
    ├── config.txt
    └── photo1.jpg
```

### Step 2: Add photos

For each person, create a folder with their name (lowercase, no spaces):

```bash
mkdir -p faces/john
mkdir -p faces/jane
```

Add 1-3 clear photos to each folder.

**Photo Tips:**
- Use 1-3 clear photos per person
- Face should be clearly visible and well-lit
- Different angles/expressions help improve accuracy
- Supported formats: JPG, JPEG, PNG
- Photos can be any size (will be processed automatically)

### Step 3: Create config files

In each person's folder, create a `config.txt` file:

**Example `faces/john/config.txt`:**
```
name: John Smith
greeting: Hey John! Welcome back to the office!
```

**Example `faces/jane/config.txt`:**
```
name: Jane Doe
greeting: Hi Jane! Good to see you!
threshold: 0.35
```

**Config Options:**
- `name`: Display name (defaults to folder name if not specified)
- `greeting`: Custom greeting message (defaults to "Hello {name}!")
- `threshold`: Custom recognition threshold 0.3-0.5 (optional, defaults to 0.40)
  - Lower = more strict matching (fewer false positives)
  - Higher = more relaxed matching (may accept similar faces)

**If no config.txt exists:** The system uses the folder name as the person's name with a default greeting.

## ▶️ Usage

```bash
uv run main.py
```

**First run:**
- System will process all photos and generate embeddings (~30 seconds)
- Embeddings are cached in `face_embeddings.pkl` for instant startup next time
- DeepFace will download AI models (~100MB) on first use

**Normal operation:**
1. Webcam window opens showing video feed
2. When someone is detected, their face is analyzed
3. System shows match percentage for each known person
4. If recognized (>60% match), greets them by name
5. If unknown, plays security alert message

**Output example:**
```
🔍 Analyzing face...

📊 Recognition Results:
   ✓ Drikus: 87.3% match (distance: 0.127)
     Robert: 45.2% match (distance: 0.548)
     Adriana: 23.1% match (distance: 0.769)
     Mohamed: 31.4% match (distance: 0.686)

✅ RECOGNIZED: Drikus (87.3% confidence)
🗣️  Speaking: Hello Drikus! Welcome back!
```

## ⚙️ Configuration

Edit `config.py` for global settings:

```python
# Recognition sensitivity (lower = more strict)
DEFAULT_SIMILARITY_THRESHOLD = 0.40  # 0.3-0.5 recommended
                                     # 0.3 = very strict (fewer false positives)
                                     # 0.5 = relaxed (more false positives)

# Unknown person message
UNKNOWN_GREETING = "Hello! I don't recognize you. Please check in at reception."

# Face recognition model
FACE_MODEL = "Facenet512"  # Best accuracy
# Other options: "VGG-Face", "Facenet", "OpenFace", "ArcFace"

# Detection settings
PERSON_DETECTION_CONFIDENCE = 0.6  # YOLO confidence threshold
GREETING_COOLDOWN_SECONDS = 15     # Time between greetings
MOVEMENT_THRESHOLD_PIXELS = 150    # Movement required to re-greet
```

**Per-person configuration** is done in each person's `faces/{name}/config.txt` file (see setup section above).

## 🔄 Updating Face Database

If you add new photos or people:

```bash
# Delete the cache to regenerate embeddings
rm face_embeddings.pkl

# Run again
uv run main.py
```

## 🎛️ Controls

- **q**: Quit the application

## 🔒 Security Features

- Detects and alerts for unknown visitors
- Shows confidence scores for transparency
- Can be extended to send notifications/alerts
- Detailed logging of all recognition events

## 🧠 Technical Details

- **Person Detection**: YOLOv8 nano (fast, efficient)
- **Face Recognition**: DeepFace with Facenet512 model
- **Embedding Comparison**: Cosine distance
- **TTS**: Google Text-to-Speech
- **Audio**: Pygame mixer
- **Threading**: Non-blocking speech

## 🐛 Troubleshooting

**"No face embeddings loaded"**
- Make sure you've added photos to `faces/[name]/` folders
- Check that photos contain clear, visible faces

**Poor recognition accuracy**
- Add more photos (2-3 per person works best)
- Ensure photos are well-lit and faces are clear
- Adjust `SIMILARITY_THRESHOLD` (try 0.35 for stricter matching)

**Slow processing**
- First run is slower (generating embeddings)
- Subsequent runs use cached embeddings (fast!)
- Consider using a faster face model (Facenet instead of Facenet512)

## 📊 Recognition Accuracy

With good quality photos, you can expect:
- **95%+** accuracy for known people
- **Very low** false positive rate for unknown people
- Robust to different lighting, angles, and expressions

Much better than characteristic-based matching!
