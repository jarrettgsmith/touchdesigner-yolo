# YOLO for TouchDesigner

Real-time object detection from TouchDesigner using Ultralytics YOLO (v8/v9/v10/v11) via Syphon (video) and OSC (detection data).

## Features

- **Syphon video I/O** - GPU texture sharing for input and annotated output
- **OSC detection data** - Bounding boxes, class labels, and confidence scores
- **Real-time performance** - MPS/CUDA GPU acceleration
- **Multiple YOLO models** - nano (fastest) to xlarge (most accurate)
- **80 COCO classes** - Person, car, dog, etc. (full COCO dataset)

## Quick Start

### 1. Setup (First Time)

```bash
chmod +x setup.sh run.sh
./setup.sh
```

This will:
- Create a virtual environment
- Install PyTorch with MPS/CUDA support
- Install Ultralytics YOLO and dependencies
- Install Syphon and OSC libraries

**Note:** First run will download the YOLO model (~6MB for nano, ~138MB for xlarge).

### 2. Run Server

```bash
./run.sh
```

### 3. TouchDesigner Setup

#### Video Input (Syphon Out TOP)
1. Create **Syphon Out TOP**
2. Set **Server Name**: `TD Video Out`
3. Connect your video source (webcam, movie, etc.)

#### Annotated Video Output (Syphon In TOP)
1. Create **Syphon In TOP**
2. Set **Server Name**: `YOLO Detections`
3. This shows video with bounding boxes and labels

#### Detection Data (OSC In CHOP)
1. Create **OSC In CHOP**
2. Set **Port**: `7000`
3. Set **Network Address**: `*`

## OSC Output Format

The server sends detection data via OSC to **port 7000** on **127.0.0.1**.

### Detection Count

```
/yolo/count <int>
```

Number of objects detected in the current frame.

### Per-Detection Data

For each detection `i` (0-indexed):

```
/yolo/<i>/class <string>        # Class name (e.g., "person", "car", "dog")
/yolo/<i>/confidence <float>    # Confidence score (0.0 - 1.0)
/yolo/<i>/bbox <x1> <y1> <x2> <y2>  # Bounding box (normalized 0.0 - 1.0)
```

**Bounding Box Coordinates:**
- `x1, y1` = top-left corner
- `x2, y2` = bottom-right corner
- All values normalized (0.0 - 1.0)

### Example OSC Messages

```
/yolo/count 3
/yolo/0/class "person"
/yolo/0/confidence 0.92
/yolo/0/bbox 0.25 0.30 0.45 0.80

/yolo/1/class "dog"
/yolo/1/confidence 0.87
/yolo/1/bbox 0.50 0.60 0.70 0.90

/yolo/2/class "car"
/yolo/2/confidence 0.95
/yolo/2/bbox 0.10 0.20 0.40 0.50
```

## Configuration

Edit `yolo_server_syphon.py`:

```python
WIDTH, HEIGHT = 1920, 1080
SYPHON_INPUT_NAME = "TD Video Out"
SYPHON_OUTPUT_NAME = "YOLO Detections"
OSC_SEND_PORT = 7000

# YOLO Configuration
YOLO_MODEL = "yolov8n.pt"  # Model selection
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence
IOU_THRESHOLD = 0.45  # Non-max suppression threshold
```

### YOLO Model Options

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolov8n.pt` | 6 MB | Fastest | Good | Real-time, limited hardware |
| `yolov8s.pt` | 22 MB | Fast | Better | Balanced performance |
| `yolov8m.pt` | 52 MB | Medium | Good | Good accuracy needed |
| `yolov8l.pt` | 88 MB | Slow | Better | High accuracy |
| `yolov8x.pt` | 138 MB | Slowest | Best | Maximum accuracy |

You can also use v9, v10, or v11 models:
```python
YOLO_MODEL = "yolov9c.pt"
YOLO_MODEL = "yolov10n.pt"
YOLO_MODEL = "yolo11n.pt"
```

### Detection Classes

YOLO detects 80 COCO classes:

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog,
horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush
```

## TouchDesigner Usage Examples

### Filter by Class

```python
# In TouchDesigner Script DAT
# Get all detections from OSC In CHOP
count = op('oscin1')['yolo/count'][0]

people = []
for i in range(int(count)):
    class_name = op('oscin1')[f'yolo/{i}/class'][0]
    if class_name == 'person':
        bbox = [
            op('oscin1')[f'yolo/{i}/bbox/0'][0],  # x1
            op('oscin1')[f'yolo/{i}/bbox/1'][0],  # y1
            op('oscin1')[f'yolo/{i}/bbox/2'][0],  # x2
            op('oscin1')[f'yolo/{i}/bbox/3'][0],  # y2
        ]
        conf = op('oscin1')[f'yolo/{i}/confidence'][0]
        people.append({'bbox': bbox, 'confidence': conf})
```

### Convert Bounding Box to Pixels

```python
# In TouchDesigner Script DAT
WIDTH, HEIGHT = 1920, 1080

# Get normalized bbox from OSC
x1_norm = op('oscin1')['yolo/0/bbox/0'][0]
y1_norm = op('oscin1')['yolo/0/bbox/1'][0]
x2_norm = op('oscin1')['yolo/0/bbox/2'][0]
y2_norm = op('oscin1')['yolo/0/bbox/3'][0]

# Convert to pixels
x1 = int(x1_norm * WIDTH)
y1 = int(y1_norm * HEIGHT)
x2 = int(x2_norm * WIDTH)
y2 = int(y2_norm * HEIGHT)

# Get center point
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

### Track Specific Objects

```python
# In TouchDesigner Script DAT
# Track cars only
count = op('oscin1')['yolo/count'][0]

cars = []
for i in range(int(count)):
    class_name = op('oscin1')[f'yolo/{i}/class'][0]
    if class_name == 'car':
        conf = op('oscin1')[f'yolo/{i}/confidence'][0]

        # Only high confidence detections
        if conf > 0.8:
            cars.append(i)

# Number of high-confidence cars
print(f"Detected {len(cars)} cars with >80% confidence")
```

## Requirements

- macOS (for Syphon)
- Python 3.10+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS)
- TouchDesigner
- ~500MB free space for models

## Performance Notes

- GPU acceleration strongly recommended for real-time performance
- Nano model runs at 30+ FPS on Apple M1/M2
- Larger models may require reducing frame rate
- MPS (Apple Silicon) provides excellent performance
- First inference is slower due to model warmup

## Troubleshooting

### No video input?

Make sure your TouchDesigner Syphon Out TOP has the exact server name `TD Video Out`

### Model download fails?

Check your internet connection. Models are downloaded from Ultralytics on first run.

### Low FPS?

Try a smaller model:
```python
YOLO_MODEL = "yolov8n.pt"  # Fastest
```

Or increase confidence threshold to reduce detections:
```python
CONFIDENCE_THRESHOLD = 0.5  # Higher = fewer detections
```

### OSC data not appearing in TouchDesigner?

1. Check OSC In CHOP port is set to `7000`
2. Verify network address is set to `*` (any)
3. Make sure objects are being detected (check Syphon output)

## Architecture

Following the [TouchDesigner External Tool Integration Pattern](../ARCHITECTURE_PATTERNS.md):

```
TouchDesigner                      YOLO Server
┌──────────────┐                  ┌──────────────┐
│ Syphon Out   │─────Video───────>│ Syphon In    │
│              │                  │              │
│ Syphon In    │<───Annotated─────│ Syphon Out   │
│              │   (boxes/labels) │              │
│ OSC In       │<───Detections────│ OSC Out      │
│ (Port 7000)  │   (bbox/class/   │              │
│              │    confidence)   │              │
└──────────────┘                  └──────────────┘
```

## Future Enhancements

- [ ] Segmentation mode (YOLOv8-seg)
- [ ] Pose estimation (YOLOv8-pose)
- [ ] Object tracking (persistent IDs)
- [ ] Custom trained models
- [ ] Multi-class filtering via OSC
- [ ] Confidence threshold control via OSC
- [ ] TouchDesigner .toe example file

## License

MIT

## Credits

- YOLO by Ultralytics: https://github.com/ultralytics/ultralytics
- TouchDesigner integration pattern inspired by depthai-handtracking
