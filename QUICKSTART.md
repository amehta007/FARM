#  Quick Start Guide


## Initial Setup

### 1. Navigate to Project Directory
```bash
cd C:\xxxxxx\xxxxx.py
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install all required packages including:
- onnxruntime (CPU inference)
- opencv-python (video processing)
- ultralytics (YOLOv8)
- streamlit (dashboard)
- and more...

### 3. Download YOLOv8 Model
```bash
python -m src.models.download_models
```

This will:
- Download YOLOv8n pretrained weights (~6MB)
- Export to ONNX format for CPU inference
- Save to `models/weights/yolov8n.onnx`

**Expected output:**
```
INFO: Downloading YOLOv8n model...
INFO: Exporting to ONNX format...
SUCCESS: Model exported to models/weights/yolov8n.onnx
```

## Running the System

### Option 1: Process the Sample Video

A synthetic sample video has already been generated at `data/raw/sample.mp4`.

```bash
python -m src.main process --video data/raw/sample.mp4 --save-annotated
```

**What this does:**
- Detects workers (people) in the video
- Tracks them with stable IDs
- Computes activity metrics (active/idle time)
- Generates occupancy heatmap
- Saves annotated video to `data/outputs/annotated_YYYYMMDD_HHMMSS.mp4`
- Saves metrics to CSV and Parquet files

**Expected output:**
```
Worker Detection & Monitoring System
============================================================
Initializing detector...
Initializing tracker...
Opening video source...
Processing started!
Processing frames...
Processed 300 frames
Processing complete!
Saving metrics...
Total tracks detected: 2
Results saved to: data/outputs/
```

### Option 2: Use Your Own Video

```bash
python -m src.main process --video path/to/your/video.mp4 --save-annotated
```

### Option 3: Use Webcam (Real-time)

```bash
python -m src.main process --webcam 0
```

- Press `q` in the video window to stop
- Change `0` to `1` or `2` if default webcam doesn't work

## View Results in Dashboard

After processing a video, launch the Streamlit dashboard:

```bash
streamlit run src/app.py
```

This will open a browser window at `http://localhost:8501` showing:
- **Overview tab**: Summary statistics, worker table, charts
- **Detailed Metrics tab**: Per-worker timeline and trajectory
- **Heatmap tab**: Spatial occupancy visualization
- **Video tab**: Playback of annotated video

## Understanding the Output

After processing, you'll find in `data/outputs/`:

1. **summary_YYYYMMDD_HHMMSS.csv**
   - Per-worker summary metrics
   - Columns: track_id, presence_time_s, active_time_s, idle_time_s, active_ratio, zone times

2. **history_YYYYMMDD_HHMMSS.parquet**
   - Frame-by-frame data for all workers
   - Includes: frame_idx, timestamp, center_x, center_y, speed_px_s, is_active

3. **heatmap_YYYYMMDD_HHMMSS.png**
   - Occupancy heatmap image (warmer = more time spent)

4. **annotated_YYYYMMDD_HHMMSS.mp4**
   - Video with bounding boxes, IDs, zones, and info panel

5. **config_YYYYMMDD_HHMMSS.yaml**
   - Configuration used for this run

## Configuration

Edit `src/configs/default.yaml` to customize:

### Adjust Activity Threshold
```yaml
metrics:
  idle_speed_px_s: 8.0  # Lower = more sensitive to activity
```

### Define Custom Zones
```yaml
zones:
  - name: "My Zone"
    points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
```

Tip: Use image editing software to find pixel coordinates.

### Speed Up Processing
```yaml
video:
  skip_frames: 3  # Process every 3rd frame (faster, less accurate)
```

### Enable Face Blurring (Privacy)
```yaml
output:
  blur_faces: true
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_metrics.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

**Note:** Some tests require full dependencies. Install everything first:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### "ONNX model not found"
Solution: Run `python -m src.models.download_models`

### "Video file not found"
Solution: Generate sample video first:
```bash
python scripts/generate_sample_video.py
```

### Processing is too slow
Solutions:
1. Increase `skip_frames` in config (e.g., 3-5)
2. Reduce video resolution before processing
3. Disable `save_annotated` if not needed

### Webcam not working on Windows
Solution: Grant camera permissions in Settings → Privacy → Camera

### "ModuleNotFoundError"
Solution: Install dependencies:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Process your own videos**: Replace `data/raw/sample.mp4` with your footage
2. **Tune parameters**: Adjust `idle_speed_px_s` threshold based on your use case
3. **Define zones**: Edit zones in config to match your farm layout
4. **Analyze results**: Use the dashboard or directly analyze CSV/Parquet files
5. **Export reports**: Download metrics from dashboard for further analysis

## Additional Commands

### Generate a new sample video
```bash
python scripts/generate_sample_video.py
```

### Run demo (all-in-one)
```bash
python -m src.main demo
```

### Process with custom config
```bash
python -m src.main process --video my_video.mp4 --config my_config.yaml
```

### Clean output directory
```bash
# Windows
rmdir /s /q data\outputs
mkdir data\outputs

# Linux/macOS
rm -rf data/outputs/*
```

## Performance Tips

For best CPU performance:
- Use `skip_frames: 2-3` for real-time monitoring
- Use `skip_frames: 1` for accurate offline analysis
- Close other heavy applications
- Process shorter videos in batches

## Support

Check the main README.md for:
- Detailed documentation
- Architecture explanation
- Privacy & ethics considerations
- Contributing guidelines
- License information

---

**You're all set! Start by running the sample video processing command above.**

