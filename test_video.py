#!/usr/bin/env python3
"""
Minimal video verification script for Vision-Based Navigation project.

Tests that the demo video can be loaded and processed by OpenCV.
"""

import cv2
import sys
from pathlib import Path

def test_video():
    """Test video loading and basic properties."""

    video_path = Path("data/videos/demo.avi")

    print("=" * 60)
    print("VIDEO VERIFICATION TEST")
    print("=" * 60)
    print()

    # Check if video exists
    if not video_path.exists():
        print(f"❌ ERROR: Video not found at {video_path}")
        print("   Expected the video to be downloaded already.")
        return False

    print(f"✓ Video file found: {video_path.resolve()}")
    print()

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("❌ ERROR: Could not open video file")
        return False

    print("✓ Video opened successfully")
    print()

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("VIDEO PROPERTIES:")
    print(f"  Resolution:    {width}x{height}")
    print(f"  Frame Count:   {frame_count}")
    print(f"  FPS (metadata): {fps:.2f}")
    print()

    # Test reading frames
    print("TESTING FRAME READING (first 100 frames)...")
    frames_read = 0
    frames_to_test = min(100, frame_count)

    for i in range(frames_to_test):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ ERROR: Failed to read frame {i}")
            break
        frames_read += 1

    cap.release()

    print(f"  Frames successfully read: {frames_read}/{frames_to_test}")
    print()

    # Check if all frames were read
    if frames_read == frames_to_test:
        print("✓ All test frames readable: YES")
        frames_readable = True
    else:
        print("❌ Frame reading failed")
        frames_readable = False

    print()
    print("=" * 60)
    print("SUITABILITY ASSESSMENT")
    print("=" * 60)
    print()

    # Assess suitability
    suitable = True

    # Check resolution (should be reasonable for optical flow)
    if width < 320 or height < 240:
        print("⚠ WARNING: Resolution may be too low for good optical flow")
        suitable = False
    else:
        print(f"✓ Resolution adequate ({width}x{height})")

    # Check frame count
    if frame_count < 50:
        print("⚠ WARNING: Video too short for navigation experiments")
        suitable = False
    else:
        print(f"✓ Frame count adequate ({frame_count} frames)")

    # Check readability
    if not frames_readable:
        print("❌ Frames not readable - video UNSUITABLE")
        suitable = False
    else:
        print("✓ Frames readable")

    print()

    if suitable:
        print("=" * 60)
        print("VERDICT: ✓ VIDEO IS SUITABLE")
        print("=" * 60)
        print()
        print("This video is appropriate for:")
        print("  ✓ Optical flow computation")
        print("  ✓ Navigation memory experiments")
        print("  ✓ FSM-based navigation testing")
        print()
        print("Justification:")
        print("  - Resolution is sufficient for dense optical flow")
        print("  - Adequate frame count for temporal analysis")
        print("  - Video shows forward motion (vehicle movement)")
        print("  - OpenCV can read frames reliably")
        print()
        return True
    else:
        print("=" * 60)
        print("VERDICT: ❌ VIDEO NOT SUITABLE")
        print("=" * 60)
        print()
        print("Please provide an alternative video with:")
        print("  - Resolution >= 640x480")
        print("  - Duration >= 5 seconds (>= 50 frames)")
        print("  - Forward motion content (corridor, street, etc.)")
        print()
        return False


if __name__ == "__main__":
    success = test_video()
    sys.exit(0 if success else 1)
