#!/usr/bin/env python3
"""
Vision-Based Navigation with Memory - Full Demo

Runs the complete navigation pipeline and produces a summary report.
This is the executable version of notebooks/05_nav_full_demo.ipynb
"""

import sys
import time
from pathlib import Path

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
    iter_frames,
    release_capture,
    draw_navigation_overlay,
)
from src.perception_flow import compute_optical_flow, compute_obstacle_risk, compute_openness
from src.memory import NavigationMemory
from src.fsm import NavigationFSM
from src.controller import NavigationController
from src.metrics import NavigationMetrics


def print_header(text, char="="):
    """Print a formatted header."""
    width = 70
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)
    print()


def main():
    """Run the full navigation demo."""

    print_header("VISION-BASED NAVIGATION WITH MEMORY - FULL DEMO", "=")

    # ===== SETUP AND CONFIGURATION =====
    print("Loading configuration...")
    config = load_config("configs/default.yaml")
    print(f"✓ Optical flow method: {config['flow']['method']}")
    print(f"✓ Risk thresholds: avoid={config['risk']['thresholds']['avoid']}, stop={config['risk']['thresholds']['stop']}")
    print(f"✓ FSM default state: {config['fsm']['default_state']}")
    print(f"✓ Recovery frames: {config['memory']['recovery_frames']}")

    # Ensure video exists
    print("\nEnsuring demo video is available...")
    video_path = ensure_demo_video(config)
    print(f"✓ Video ready: {video_path.resolve()}")

    # ===== INITIALIZE NAVIGATION SYSTEM =====
    print_header("INITIALIZING NAVIGATION SYSTEM")

    memory = NavigationMemory(config)
    fsm = NavigationFSM(config)
    controller = NavigationController(config)
    metrics = NavigationMetrics()

    print("✓ Navigation system initialized:")
    print(f"  - Memory: risk_window={config['risk']['risk_smooth_window']}, cooldown={config['memory']['cooldown_frames']}")
    print(f"  - FSM: {fsm.default_state} → transitions based on risk/openness")
    print(f"  - Controller: linear_v={controller.linear_v}, angular_w={controller.angular_w}")
    print(f"  - Metrics: tracking FPS, latency, and state durations")

    # ===== RUN FULL NAVIGATION LOOP =====
    print_header("RUNNING FULL NAVIGATION LOOP")

    cap = open_capture(config)
    max_frames = config["video"].get("max_frames") or 300

    # Start metrics
    metrics.start()

    # Storage for sample frames
    sample_frames = []
    sample_indices = []
    sample_interval = 30

    prev_gray = None
    process_count = 0

    print(f"Processing up to {max_frames} frames...")
    print("Pipeline: Optical Flow → Risk/Openness → Memory → FSM → Controller → Metrics")
    print()

    for idx, frame in tqdm(iter_frames(cap, max_frames=max_frames, convert_rgb=False),
                           total=max_frames, desc="Processing"):
        frame_start = time.perf_counter()

        # Convert to grayscale for optical flow
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # PERCEPTION: Compute optical flow and extract signals
            flow = compute_optical_flow(prev_gray, curr_gray, config)
            risk_raw = compute_obstacle_risk(flow, config)
            openness_raw, direction = compute_openness(flow, config)

            # MEMORY: Apply temporal smoothing and hysteresis
            signals = memory.update(risk_raw, openness_raw, direction)

            # FSM: Determine navigation state
            state = fsm.update(signals)

            # CONTROLLER: Generate motion command
            recovery_dir = fsm.get_recovery_direction()
            preferred_dir = recovery_dir if recovery_dir else signals["preferred_direction"]
            command = controller.get_command(state, preferred_dir)

            # Compute frame processing latency
            frame_end = time.perf_counter()
            latency = frame_end - frame_start

            # METRICS: Record performance
            metrics.record(state, latency)

            # Store sample frame info (for later analysis)
            if idx % sample_interval == 0:
                sample_indices.append(idx)

            process_count += 1

        prev_gray = curr_gray

    release_capture(cap)

    print()
    print(f"✓ Processing complete: {process_count} frames")

    # ===== PERFORMANCE METRICS SUMMARY =====
    print_header("PERFORMANCE METRICS SUMMARY")

    metrics.print_summary()

    # ===== DETAILED ANALYSIS =====
    print_header("DETAILED ANALYSIS")

    summary = metrics.summary()
    perf = summary["performance"]
    states = summary["states"]

    # State percentages
    print("State Distribution:")
    state_percentages = states["state_percentages"]
    for state, pct in sorted(state_percentages.items(), key=lambda x: -x[1]):
        bar_length = int(pct / 2)  # Scale to fit in terminal
        bar = "█" * bar_length
        print(f"  {state:20s} {pct:5.1f}% {bar}")

    print()
    print(f"State Transitions: {states['state_transitions']}")
    print(f"Recovery Events:   {states['recovery_events']}")

    # Latency percentiles
    print()
    print("Latency Distribution:")
    latency_stats = perf["latency_ms"]
    print(f"  Mean:  {latency_stats['mean']:6.2f} ms")
    print(f"  P50:   {latency_stats['p50']:6.2f} ms")
    print(f"  P95:   {latency_stats['p95']:6.2f} ms")
    print(f"  P99:   {latency_stats['p99']:6.2f} ms")

    # ===== FINAL SUMMARY REPORT =====
    print_header("FINAL SUMMARY REPORT", "=")

    print("System Components:")
    print("  ✓ Optical Flow Perception (Mode A)")
    print("  ✓ Temporal Memory with Hysteresis")
    print("  ✓ Finite State Machine with Anti-Oscillation")
    print("  ✓ Motion Command Controller")
    print("  ✓ Real-time Performance Monitoring")
    print()

    print("Key Results:")
    print(f"  Frames Processed:     {perf['frames_processed']}")
    print(f"  Average FPS:          {perf['fps']:.2f}")
    print(f"  Mean Latency:         {perf['latency_ms']['mean']:.2f} ms")
    print(f"  P95 Latency:          {perf['latency_ms']['p95']:.2f} ms")
    print(f"  State Transitions:    {states['state_transitions']}")
    print(f"  Recovery Events:      {states['recovery_events']}")
    print()

    # System validation
    meets_fps = perf['fps'] >= 5.0
    meets_latency = perf['latency_ms']['p95'] < 500
    has_states = len(state_percentages) >= 1

    print("System Validation:")
    print(f"  ✓ FPS acceptable (>= 5 FPS):         {'PASS' if meets_fps else 'FAIL'}")
    print(f"  ✓ Latency acceptable (P95 < 500ms):  {'PASS' if meets_latency else 'FAIL'}")
    print(f"  ✓ FSM operational:                    {'PASS' if has_states else 'FAIL'}")
    print()

    if meets_fps and meets_latency and has_states:
        print("=" * 70)
        print("  ✓✓✓  DEMO SUCCESSFUL - ALL SYSTEMS OPERATIONAL  ✓✓✓")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("  ⚠ System may need tuning for optimal performance")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
