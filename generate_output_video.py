#!/usr/bin/env python3
"""
Generate output video with navigation overlay.

Creates a video file showing the navigation system in action with:
- Real-time state display
- Motion commands
- Risk and openness scores
- FPS and latency metrics
- Visual overlays
"""

import sys
import time
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from tqdm import tqdm

# Project modules
from src.utils import (
    load_config,
    ensure_demo_video,
    open_capture,
    release_capture,
    draw_navigation_overlay,
)
from src.perception_flow import compute_optical_flow, compute_obstacle_risk, compute_openness
from src.memory import NavigationMemory
from src.fsm import NavigationFSM
from src.controller import NavigationController
from src.metrics import NavigationMetrics


def generate_output_video(
    output_path: str = "output/nav_demo_output.mp4",
    max_frames: int = None,
    fps: int = 10,
    codec: str = "mp4v",
):
    """
    Generate output video with navigation overlay.

    Args:
        output_path: Path to save output video
        max_frames: Maximum frames to process (None = all)
        fps: Output video FPS
        codec: Video codec (mp4v, avc1, xvid)
    """

    print("=" * 70)
    print("  GENERATING NAVIGATION OUTPUT VIDEO")
    print("=" * 70)
    print()

    # Load configuration
    print("Loading configuration...")
    config = load_config("configs/default.yaml")

    # Override max_frames if specified
    if max_frames is None:
        max_frames = config["video"].get("max_frames") or 300

    # Ensure video exists
    video_path = ensure_demo_video(config)
    print(f"✓ Input video: {video_path}")

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output video: {output_path.resolve()}")
    print()

    # Initialize navigation system
    print("Initializing navigation system...")
    memory = NavigationMemory(config)
    fsm = NavigationFSM(config)
    controller = NavigationController(config)
    metrics = NavigationMetrics()
    metrics.start()
    print("✓ All components ready")
    print()

    # Open input video
    cap = open_capture(config)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Output FPS: {fps}")
    print(f"  Codec: {codec}")
    print()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer with codec {codec}")

    print(f"Processing up to {max_frames} frames...")
    print("Pipeline: Flow → Risk/Openness → Memory → FSM → Controller → Overlay → Write")
    print()

    prev_gray = None
    frame_count = 0
    written_count = 0

    # Process frames
    for idx in tqdm(range(max_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.perf_counter()

        # Convert to grayscale for optical flow
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # PERCEPTION
            flow = compute_optical_flow(prev_gray, curr_gray, config)
            risk_raw = compute_obstacle_risk(flow, config)
            openness_raw, direction = compute_openness(flow, config)

            # MEMORY
            signals = memory.update(risk_raw, openness_raw, direction)

            # FSM
            state = fsm.update(signals)

            # CONTROLLER
            recovery_dir = fsm.get_recovery_direction()
            preferred_dir = recovery_dir if recovery_dir else signals["preferred_direction"]
            command = controller.get_command(state, preferred_dir)

            # METRICS
            frame_end = time.perf_counter()
            latency = frame_end - frame_start
            metrics.record(state, latency)

            # OVERLAY
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_fps = metrics.perf.get_fps()
            frame_overlay = draw_navigation_overlay(
                frame_rgb,
                state=state,
                command_name=command.command_name,
                risk=signals["risk_smooth"],
                openness=signals["openness_smooth"],
                fps=current_fps,
                latency_ms=latency * 1000,
            )

            # Convert back to BGR for video writing
            frame_bgr = cv2.cvtColor(frame_overlay, cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(frame_bgr)
            written_count += 1

        prev_gray = curr_gray
        frame_count += 1

    # Release resources
    release_capture(cap)
    out.release()

    print()
    print("=" * 70)
    print("  VIDEO GENERATION COMPLETE")
    print("=" * 70)
    print()

    # Print summary
    summary = metrics.summary()
    perf = summary["performance"]
    states = summary["states"]

    print(f"Output video saved: {output_path.resolve()}")
    print()
    print(f"Statistics:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Frames written:   {written_count}")
    print(f"  Average FPS:      {perf['fps']:.2f}")
    print(f"  Mean latency:     {perf['latency_ms']['mean']:.2f} ms")
    print()
    print(f"State Distribution:")
    for state, pct in sorted(states["state_percentages"].items(), key=lambda x: -x[1]):
        print(f"  {state:20s}: {pct:5.1f}%")
    print()
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Verify output
    verify_cap = cv2.VideoCapture(str(output_path))
    if verify_cap.isOpened():
        verify_count = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        verify_cap.release()
        print(f"✓ Output video verified: {verify_count} frames readable")
    else:
        print("⚠ Warning: Could not verify output video")

    print()
    print("To play the video:")
    print(f"  ffplay {output_path}")
    print(f"  vlc {output_path}")
    print(f"  mpv {output_path}")
    print()

    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate navigation output video with overlay"
    )
    parser.add_argument(
        "-o", "--output",
        default="output/nav_demo_output.mp4",
        help="Output video path (default: output/nav_demo_output.mp4)"
    )
    parser.add_argument(
        "-n", "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: 300 or config value)"
    )
    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=10,
        help="Output video FPS (default: 10)"
    )
    parser.add_argument(
        "-c", "--codec",
        default="mp4v",
        choices=["mp4v", "avc1", "xvid", "h264"],
        help="Video codec (default: mp4v)"
    )

    args = parser.parse_args()

    try:
        output_path = generate_output_video(
            output_path=args.output,
            max_frames=args.max_frames,
            fps=args.fps,
            codec=args.codec,
        )
        print("✓ Success!")
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
