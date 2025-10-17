# Face Photos Directory

Add reference photos for each person here. Each person gets their own folder with photos and an optional config file.

## Structure

```
faces/
├── person.config.example   # Example config file (copy this!)
├── john/
│   ├── config.txt         # John's personal config
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── jane/
│   ├── config.txt         # Jane's personal config
│   ├── photo1.jpg
│   └── photo2.jpg
└── alice/
    └── photo1.jpg         # No config.txt = uses folder name
```

## Quick Setup

1. **Create a folder** (lowercase, no spaces):
   ```bash
   mkdir -p faces/yourname
   ```

2. **Add photos** (1-3 photos recommended):
   ```bash
   # Copy your photos to the folder
   cp photo1.jpg faces/yourname/
   ```

3. **Create config.txt** (optional but recommended):
   ```bash
   # Copy the example
   cp faces/person.config.example faces/yourname/config.txt
   
   # Edit it with your info
   nano faces/yourname/config.txt
   ```

## Config File Format

Create `config.txt` in each person's folder:

```
name: John Smith
greeting: Hey John! Welcome back!
threshold: 0.35
```

**Options:**
- `name`: Display name (optional, defaults to folder name)
- `greeting`: Custom greeting (optional, defaults to "Hello {name}!")
- `threshold`: Recognition strictness 0.3-0.5 (optional, defaults to 0.40)
  - Lower = more strict (0.3 = very strict)
  - Higher = more lenient (0.5 = relaxed)

**Note:** All config options are optional! If you don't create a config.txt, the system will use sensible defaults based on the folder name.

## Quick Capture

Use the capture tool to easily capture photos from your webcam:

```bash
uv run capture_faces.py
```

## Manual Addition

1. Create a folder for each person (lowercase name)
2. Add 1-3 clear photos of their face
3. Supported formats: JPG, JPEG, PNG
4. Photos should be well-lit with face clearly visible

## Tips for Best Results

- **Lighting**: Good, even lighting works best
- **Angles**: Straight-on face photos are ideal
- **Multiple photos**: 2-3 photos per person improves accuracy
- **Variety**: Different expressions/angles help
- **Quality**: Clear, in-focus photos work better
- **Consistency**: If you wear glasses daily, include them in photos

## After Adding Photos

Run the main application to generate embeddings:

```bash
uv run main.py
```

The system will process all photos and cache the face embeddings.
