# Vision-Based Navigation with Memory

Camera-only navigation system with optical flow perception, temporal memory, and finite-state machine control. Processes video frames to estimate obstacle risk and issue simulated robot commands with anti-oscillation behavior.

---

## Features

- ✅ **Optical Flow Perception** - Dense flow computation for obstacle risk estimation
- ✅ **Temporal Memory** - Rolling buffers with hysteresis for signal smoothing
- ✅ **Finite State Machine** - Navigation control with anti-oscillation recovery
- ✅ **Real-time Performance** - 10-15 FPS processing on CPU
- ✅ **Rich Visualization** - Overlay with state, commands, risk, and metrics
- ✅ **Video Output** - Generate annotated videos with navigation overlay
- ✅ **Comprehensive Metrics** - FPS, latency, and state duration tracking

---

## Quick Start

### Option 1: Run Complete Demo (Command-Line)
```bash
# Install dependencies
pip install -r requirements.txt

# Run full demo (processes 300 frames, shows metrics)
python3 run_demo.py
```

### Option 2: Generate Output Video
```bash
# Generate video with navigation overlay (saves to output/)
python3 generate_output_video.py --max-frames 200

# Play the output video
ffplay output/nav_demo_output.mp4
```

### Option 3: Interactive Notebooks
```bash
# Run Jupyter and open notebooks
jupyter notebook

# Navigate to notebooks/ and run:
# - 00_setup_and_video_io.ipynb (video verification)
# - 05_nav_full_demo.ipynb (complete demo)
```

---

## Project Structure

```
vision_based_navigation/
├── configs/
│   └── default.yaml              # System configuration
├── data/
│   └── videos/
│       └── demo.avi              # Auto-downloaded demo video
├── notebooks/
│   ├── 00_setup_and_video_io.ipynb
│   ├── 01_optical_flow_basics.ipynb
│   ├── 02_risk_scoring_and_memory.ipynb
│   ├── 03_fsm_navigation.ipynb
│   ├── 04_overlay_and_metrics.ipynb
│   └── 05_nav_full_demo.ipynb   # Final integrated demo
├── output/
│   └── nav_demo_output.mp4       # Generated output video
├── src/
│   ├── perception_flow.py        # Optical flow & risk scoring
│   ├── memory.py                 # Temporal smoothing & hysteresis
│   ├── fsm.py                    # Finite state machine
│   ├── controller.py             # Motion commands
│   ├── metrics.py                # Performance tracking
│   └── utils.py                  # Config, I/O, visualization
├── run_demo.py                   # Standalone demo script
├── generate_output_video.py      # Video generation script
├── test_video.py                 # Video verification
└── requirements.txt              # Python dependencies
```

---

## Usage

### 1. Video Verification
Test that the demo video is properly downloaded and readable:
```bash
python3 test_video.py
```

### 2. Run Navigation Demo
Execute the full navigation pipeline and view metrics:
```bash
python3 run_demo.py
```

**Output:**
- Processes video frames through complete pipeline
- Displays performance metrics (FPS, latency)
- Shows state distribution
- Validates system functionality

### 3. Generate Annotated Video
Create a video file with navigation overlays:
```bash
# Basic usage (200 frames, 10 FPS)
python3 generate_output_video.py

# Custom options
python3 generate_output_video.py \
  --max-frames 300 \
  --fps 10 \
  --output output/my_demo.mp4 \
  --codec mp4v
```

**Output:**
- Video file with navigation state overlay
- Risk visualization bar
- Command and metrics display
- Saves to `output/` directory

**Options:**
- `-o, --output PATH` - Output video path
- `-n, --max-frames N` - Max frames to process
- `-f, --fps N` - Output video frame rate
- `-c, --codec CODEC` - Video codec (mp4v, avc1, xvid)

### 4. Interactive Notebooks
Run Jupyter notebooks for step-by-step exploration:
```bash
jupyter notebook
```

**Notebook Phases:**
1. **Phase 1** (00): Video I/O and frame reading
2. **Phase 2** (01): Optical flow computation
3. **Phase 3** (02): Memory and smoothing
4. **Phase 4** (03): FSM navigation
5. **Phase 5** (04): Overlay and metrics
6. **Phase 6** (05): Final integrated demo

---

## System Architecture

```
Video Frame (BGR)
    ↓
Optical Flow (Farneback)
    ↓
Risk & Openness Signals
    ↓
Temporal Memory (Rolling Buffer + Hysteresis)
    ↓
Finite State Machine (CRUISE/AVOID/STOP/RECOVERY)
    ↓
Motion Controller (v, ω commands)
    ↓
Metrics & Visualization
```

---

## Navigation States

The FSM operates in the following states:

| State | Description | Command | Trigger |
|-------|-------------|---------|---------|
| **CRUISE_FORWARD** | Normal forward motion | v=0.35, ω=0 | Risk < avoid threshold |
| **AVOID_LEFT** | Turn left to avoid | v=0.175, ω=0.7 | Risk ≥ avoid, left more open |
| **AVOID_RIGHT** | Turn right to avoid | v=0.175, ω=-0.7 | Risk ≥ avoid, right more open |
| **STOP_TOO_CLOSE** | Emergency stop | v=0, ω=0 | Risk ≥ stop threshold |
| **RECOVERY** | Anti-oscillation mode | v=0.105, ω=±0.7 | Oscillation detected |

---

## Configuration

Edit `configs/default.yaml` to customize:

### Key Parameters

```yaml
risk:
  thresholds:
    avoid: 0.55    # Trigger avoidance behavior
    stop: 0.8      # Trigger emergency stop

memory:
  cooldown_frames: 8      # Prevent rapid direction changes
  recovery_frames: 18     # Duration of recovery mode

flow:
  method: farneback       # Optical flow algorithm
```

### Tuning Tips

- **Lower avoid threshold** (e.g., 0.45) → More cautious behavior
- **Higher recovery frames** (e.g., 25) → Stronger oscillation prevention
- **Larger window size** → Smoother but slower response

---

## Performance

### Benchmark Results (demo video, 299 frames)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **FPS** | 13.08 | ≥ 5.0 | ✅ PASS |
| **Mean Latency** | 75.64 ms | < 200 ms | ✅ PASS |
| **P95 Latency** | 81.40 ms | < 500 ms | ✅ PASS |

### System Requirements

- **Python:** 3.10+
- **CPU:** Any modern processor (no GPU required)
- **Memory:** ~200 MB
- **Disk:** ~20 MB (code + demo video)

---

## Output Examples

### Console Output (run_demo.py)
```
======================================================================
  VISION-BASED NAVIGATION WITH MEMORY - FULL DEMO
======================================================================

Performance:
  Frames processed: 299
  Average FPS: 13.08

State Distribution:
  CRUISE_FORWARD      :  89.0%
  RECOVERY            :  11.0%

✓✓✓  DEMO SUCCESSFUL - ALL SYSTEMS OPERATIONAL  ✓✓✓
```

### Generated Video
- **File:** `output/nav_demo_output.mp4` (3.3 MB)
- **Content:** Original video with overlay showing:
  - Navigation state (color-coded)
  - Motion command
  - Risk score with bar
  - Openness metric
  - Real-time FPS and latency

---

## Troubleshooting

### Video not downloading
The system auto-downloads from: `https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi`

If download fails:
```bash
# Manual download
mkdir -p data/videos
wget -O data/videos/demo.avi https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi
```

### Import errors
```bash
# Ensure you're in the project root
cd /path/to/vision_based_navigation

# Run with Python path
PYTHONPATH=. python3 run_demo.py
```

### Video writer fails
Try different codec:
```bash
python3 generate_output_video.py --codec xvid
```

---

## Documentation

- **SESSION_LOG.md** - Complete implementation history
- **TEST_REPORT.md** - Video verification results
- **DEMO_RESULTS.md** - Demo execution results
- **OUTPUT_VIDEO_GUIDE.md** - Video generation guide
- **CLAUDE.md** - Project constraints and rules
- **start_prompt.md** - Implementation requirements

---

## Development

### Running Tests
```bash
# Video verification
python3 test_video.py

# Visual inspection
jupyter notebook test_video_visual.ipynb
```

### Adding Custom Videos
1. Place video in `data/videos/`
2. Update `configs/default.yaml`:
   ```yaml
   video:
     source: data/videos/your_video.mp4
   ```
3. Run demo

### Extending the System

**Mode B (YOLO Detection):**
- Placeholder exists in `src/perception_yolo.py`
- Install: `pip install ultralytics`
- Implement detection-based risk scoring

**ROS2 Integration:**
- Convert commands to Twist messages
- Subscribe to camera topics
- Publish velocity commands

---

## Citation

If you use this project in your research, please cite:

```
Vision-Based Navigation with Memory
Camera-only Autonomy Demo
2026
```

---

## License

This project is provided for educational and research purposes.

---

## Acknowledgments

- OpenCV for optical flow algorithms
- Demo video from OpenCV samples (vtest.avi)

---

## Contact & Support

For issues or questions:
1. Check documentation in this repository
2. Review `SESSION_LOG.md` for implementation details
3. See `TROUBLESHOOTING` section above

---

**Status:** ✅ Production Ready
**Version:** 1.0
**Last Updated:** 2026-01-04
