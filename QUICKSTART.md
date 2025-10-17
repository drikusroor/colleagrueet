# Quick Start Guide

## First Time Setup

1. **Add your face photos:**
   ```bash
   # Create your folder
   mkdir -p faces/yourname
   
   # Add photos (use webcam capture tool or copy existing photos)
   uv run capture_faces.py
   # OR
   cp /path/to/your/photo.jpg faces/yourname/photo1.jpg
   ```

2. **Create your config** (optional but recommended):
   ```bash
   # Copy the example
   cp faces/person.config.example faces/yourname/config.txt
   
   # Edit with your info
   nano faces/yourname/config.txt
   ```
   
   Example config:
   ```
   name: John Smith
   greeting: Hey John! Welcome to the office!
   ```

3. **Run the greeter:**
   ```bash
   uv run main.py
   ```
   
   First run will:
   - Process all face photos (~30 seconds)
   - Download AI models (~100MB, one-time)
   - Cache embeddings for fast future startups

## Adding More People

Just repeat step 1 & 2 for each person:

```bash
mkdir -p faces/colleague
cp photo.jpg faces/colleague/photo1.jpg
echo "name: My Colleague\ngreeting: Hi there!" > faces/colleague/config.txt
```

Then delete the cache and restart:
```bash
rm face_embeddings.pkl
uv run main.py
```

## Configuration Files (Privacy)

The `config.txt` files in each person's folder are **ignored by git** for privacy. This means:
- ✅ Code and structure can be shared
- ✅ Photos are private (also gitignored)
- ✅ Names and greetings are private
- ✅ Each user configures their own setup

## Tips

- **2-3 photos per person** works best
- **Well-lit, clear faces** improve accuracy
- **Different angles/expressions** help robustness
- **No config.txt needed** - defaults work fine!
- **Custom thresholds** can be set per-person if needed

## Example Folder Structure

```
faces/
├── person.config.example
├── alice/
│   ├── config.txt (gitignored)
│   ├── photo1.jpg (gitignored)
│   └── photo2.jpg (gitignored)
└── bob/
    └── photo1.jpg (gitignored)
    # No config.txt = uses defaults
```
