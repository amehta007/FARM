# 👷 AI-Powered Worker Monitoring System

A CPU-optimized, laptop-ready MVP system for detecting and tracking workers in farm videos, computing productivity metrics, and visualizing results in an interactive dashboard.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.11-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## 🎯 Overview

This system provides:
- **Real-time person detection** using YOLOv8n (ONNX Runtime on CPU)
- **Multi-object tracking** with ByteTrack for stable ID assignment
- **Activity metrics** (active/idle time based on movement speed)
- **Zone analytics** (time spent in user-defined polygon zones)
- **Occupancy heatmaps** for spatial visualization
- **Interactive Streamlit dashboard** for analysis and reporting

**Key Features:**
- ✅ Runs on CPU only (no GPU required)
- ✅ Works with video files or live webcam
- ✅ Optimized for typical developer laptops (Windows/macOS/Linux)
- ✅ Privacy-first (local processing only, optional face blurring)
- ✅ Reproducible one-command setup

## 📸 Screenshots

*[Screenshot placeholders - run the system to generate actual screenshots]*

- Dashboard Overview
- Activity Timeline
- Occupancy Heatmap
- Live Detection Feed

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- 4GB+ RAM
- Webcam (optional, for live monitoring)

### Installation

```bash
# Clone or extract the repository
cd worker-detector

# Install dependencies
pip install -r requirements.txt

# Download and export YOLOv8n model to ONNX
python -m src.models.download_models
```

### Generate Sample Video (for testing)

```bash
# Generate a 10-second synthetic video
python scripts/generate_sample_video.py
```

### Run Detection

```bash
# Process a video file
python -m src.main process --video data/raw/sample.mp4 --save-annotated

# Or use webcam
python -m src.main process --webcam 0 --save-annotated

# With custom config
python -m src.main process --video my_video.mp4 --config src/configs/default.yaml
```

### Launch Dashboard

```bash
streamlit run src/app.py
```

Open your browser to `http://localhost:8501` to view the dashboard.

## 📁 Project Structure

```
worker-detector/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── Makefile                    # Convenience targets
├── src/
│   ├── app.py                 # Streamlit dashboard
│   ├── main.py                # CLI application
│   ├── configs/
│   │   └── default.yaml       # Configuration file
│   ├── models/
│   │   └── download_models.py # Model download/export
│   ├── detection/
│   │   └── detector.py        # ONNX inference
│   ├── tracking/
│   │   └── bytetrack.py       # Multi-object tracking
│   ├── analytics/
│   │   ├── zones.py           # Zone management
│   │   ├── metrics.py         # Activity metrics
│   │   └── heatmap.py         # Occupancy heatmap
│   ├── io/
│   │   ├── video_reader.py    # Video/webcam input
│   │   └── sink.py            # Output writers
│   ├── vis/
│   │   └── draw.py            # Visualization utilities
│   └── utils/
│       ├── settings.py        # Config management
│       ├── geometry.py        # Geometry utilities
│       └── timing.py          # FPS counter
├── data/
│   ├── raw/                   # Input videos
│   └── outputs/               # Results (CSV, videos, heatmaps)
├── models/
│   └── weights/               # ONNX model files
├── tests/                     # Unit tests
└── scripts/                   # Demo and utility scripts
```

## ⚙️ Configuration

Edit `src/configs/default.yaml` to customize:

### Video Source
```yaml
video:
  source: "data/raw/sample.mp4"  # or "webcam:0"
  skip_frames: 2                 # Process every Nth frame
```

### Detection Settings
```yaml
model:
  conf_threshold: 0.25    # Minimum confidence
  iou_threshold: 0.45     # NMS IoU threshold
```

### Zones
Define polygon zones for spatial analytics:
```yaml
zones:
  - name: "Row A"
    points: [[50, 50], [600, 50], [600, 300], [50, 300]]
  - name: "Row B"
    points: [[50, 320], [600, 320], [600, 600], [50, 600]]
```

### Activity Metrics
```yaml
metrics:
  idle_speed_px_s: 8.0    # Speed threshold for idle detection
  heatmap_grid: [48, 27]  # Heatmap grid resolution
```

## 📊 Metrics Explained

### Per-Worker Metrics

1. **Presence Time**: Total time worker is visible in frame
2. **Active Time**: Time when worker is moving (speed > threshold)
3. **Idle Time**: Time when worker is stationary (speed < threshold)
4. **Active Ratio**: Active time / Presence time
5. **Zone Time**: Time spent in each defined zone

### Speed-Based Activity Detection

The system uses a simple but effective heuristic:
- Compute center point of each detection
- Calculate instantaneous speed between frames: `speed = distance / time`
- Compare speed to threshold (default: 8 px/s)
- Classify as **active** (moving) or **idle** (stationary)

**Tuning the threshold:**
- Higher values → more sensitive (classify more as idle)
- Lower values → less sensitive (classify more as active)
- Adjust based on camera angle, resolution, and worker activity patterns

### Heatmap

The occupancy heatmap shows:
- Where workers spend most time (hot spots)
- Spatial distribution of activity
- Potential bottlenecks or underutilized areas

## 🎮 Usage Examples

### Process Multiple Videos

```bash
for video in data/raw/*.mp4; do
    python -m src.main process --video "$video" --save-annotated
done
```

### Webcam Monitoring (Real-time)

```bash
python -m src.main process --webcam 0 --display
```

Press `q` in the video window to stop.

### Export Metrics for Analysis

After processing, find in `data/outputs/`:
- `summary_YYYYMMDD_HHMMSS.csv` - Per-worker summary
- `history_YYYYMMDD_HHMMSS.parquet` - Frame-by-frame data
- `heatmap_YYYYMMDD_HHMMSS.png` - Heatmap image
- `annotated_YYYYMMDD_HHMMSS.mp4` - Annotated video

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run specific test:

```bash
pytest tests/test_detector.py -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Troubleshooting

### Model Download Issues

If `download_models.py` fails:
1. Check internet connection
2. Manually download YOLOv8n: `https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt`
3. Export using Ultralytics: `yolo export model=yolov8n.pt format=onnx`
4. Place `yolov8n.onnx` in `models/weights/`

### Slow Performance

If processing is too slow:
1. Increase `skip_frames` in config (e.g., 3-5)
2. Reduce video resolution before processing
3. Lower `conf_threshold` to reduce false positives
4. Disable `save_annotated` if not needed

### Webcam Not Found (Windows)

Grant camera permissions:
1. Settings → Privacy → Camera
2. Enable "Allow apps to access your camera"
3. Try different device indices: `--webcam 1`, `--webcam 2`

### Webcam Not Found (macOS)

Grant terminal camera access:
1. System Preferences → Security & Privacy → Camera
2. Check the box for Terminal or your IDE

## 📝 Limitations & Future Work

### Current Limitations

1. **Simple activity heuristic**: Speed-based classification may not capture all activity types (e.g., bent-over work with little displacement)
2. **No re-identification**: Lost tracks receive new IDs after occlusion
3. **2D analysis only**: No depth estimation or 3D metrics
4. **Person class only**: Doesn't distinguish roles or use tools/equipment

### Potential Improvements

- [ ] Lightweight pose estimation for better activity classification
- [ ] Re-ID module to recover IDs after occlusion
- [ ] Calibration (px→meters) for real-world speed/distance
- [ ] Multi-camera support with cross-camera tracking
- [ ] Export time-series reports (hourly/daily summaries)
- [ ] Anomaly detection (e.g., prolonged idle, unexpected absence)
- [ ] Mobile app for field supervisors

## 🔒 Privacy & Ethics

This system is designed with privacy in mind:

- ✅ **Local processing only** - No cloud uploads or external API calls
- ✅ **Optional face blurring** - Enable `blur_faces: true` in config
- ✅ **Transparent metrics** - All computations are explainable and auditable
- ⚠️ **Ethical use**: This tool is for productivity insights, NOT surveillance. Ensure:
  - Workers are informed and consent to monitoring
  - Data is used fairly and transparently
  - Metrics support workers, not replace them
  - Comply with local labor laws and privacy regulations

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Improved activity classifiers
- Additional visualization options
- Performance optimizations
- Better re-identification
- Multi-camera support

## 📄 License

This project uses:
- **YOLOv8** (Ultralytics) - AGPL-3.0 (weights) / GPL-3.0 (code)
- **ByteTrack** - MIT-like permissive
- Other dependencies - See `requirements.txt`

Ensure compliance with Ultralytics license if using commercially.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Fast object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Robust tracking
- [ONNX Runtime](https://onnxruntime.ai/) - Efficient CPU inference
- [Streamlit](https://streamlit.io/) - Beautiful dashboards

## 📞 Support

For issues or questions:
1. Check existing issues in the repository
2. Review troubleshooting section above
3. Open a new issue with:
   - System info (OS, Python version)
   - Error message or unexpected behavior
   - Steps to reproduce

---

**Built with ❤️ for reliable, CPU-friendly worker monitoring**


