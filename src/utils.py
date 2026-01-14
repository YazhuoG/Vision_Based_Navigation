import os
from pathlib import Path
from typing import Dict, Generator, Tuple, Optional
import urllib.request

import cv2
import yaml


def load_config(config_path: str = "configs/default.yaml") -> Dict:
    """Load YAML config into a dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory for the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, overwrite: bool = False) -> Path:
    """
    Download a file from url to dest.
    Returns dest path. Skips download if file exists unless overwrite is True.
    """
    ensure_parent_dir(dest)
    if dest.exists() and not overwrite:
        return dest

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    urllib.request.urlretrieve(url, tmp_path)  # nosec - trusted URL provided in constraints
    tmp_path.replace(dest)
    return dest


def ensure_demo_video(config: Dict) -> Path:
    """Ensure the demo video exists, downloading from the approved URL if needed."""
    video_path = Path(config["video"]["source"])
    if video_path.exists():
        return video_path

    url = config["video"]["backup_url"]
    return download_file(url, video_path, overwrite=False)


def open_capture(config: Dict) -> cv2.VideoCapture:
    """
    Open a cv2.VideoCapture based on config.
    If use_webcam is true, open the webcam index; otherwise open the configured video path.
    """
    if config["video"].get("use_webcam", False):
        cap = cv2.VideoCapture(int(config["video"].get("webcam_index", 0)))
    else:
        cap = cv2.VideoCapture(str(config["video"]["source"]))
    return cap


def iter_frames(
    cap: cv2.VideoCapture,
    max_frames: Optional[int] = None,
    convert_rgb: bool = True,
) -> Generator[Tuple[int, Optional[object]], None, None]:
    """
    Generator over frames from an open VideoCapture.
    Yields (index, frame) where frame is None if capture fails.
    """
    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if convert_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield idx, frame
        idx += 1


def release_capture(cap: cv2.VideoCapture) -> None:
    """Release a VideoCapture safely."""
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()


def draw_navigation_overlay(
    frame,
    state: str,
    command_name: str,
    risk: float,
    openness: float,
    fps: float = None,
    latency_ms: float = None,
):
    """
    Draw navigation information overlay on frame.

    Args:
        frame: Frame to draw on (will be modified in place)
        state: FSM state name
        command_name: Command being issued
        risk: Risk score [0, 1]
        openness: Openness score
        fps: Optional FPS value
        latency_ms: Optional latency in milliseconds

    Returns:
        frame: Frame with overlay (same object, modified)
    """
    import numpy as np

    h, w = frame.shape[:2]

    # Define colors
    color_text = (255, 255, 255)
    color_state = (0, 255, 255)  # Cyan
    color_command = (0, 255, 0)  # Green
    color_risk_low = (0, 255, 0)
    color_risk_med = (0, 165, 255)
    color_risk_high = (0, 0, 255)

    # Draw semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (400, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Text position
    y_pos = 25
    x_pos = 15
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Draw state
    cv2.putText(frame, f"State: {state}", (x_pos, y_pos), font, font_scale, color_state, thickness)
    y_pos += 25

    # Draw command
    cv2.putText(frame, f"Command: {command_name}", (x_pos, y_pos), font, font_scale, color_command, thickness)
    y_pos += 25

    # Draw risk with color coding
    risk_color = color_risk_low
    if risk > 0.8:
        risk_color = color_risk_high
    elif risk > 0.55:
        risk_color = color_risk_med
    cv2.putText(frame, f"Risk: {risk:.3f}", (x_pos, y_pos), font, font_scale, risk_color, thickness)
    y_pos += 25

    # Draw openness
    openness_text = f"Openness: {openness:.3f}"
    if openness > 0:
        openness_text += " (L>R)"
    else:
        openness_text += " (R>L)"
    cv2.putText(frame, openness_text, (x_pos, y_pos), font, font_scale, color_text, thickness)
    y_pos += 25

    # Draw FPS if provided
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (x_pos, y_pos), font, font_scale, color_text, thickness)
        y_pos += 25

    # Draw latency if provided
    if latency_ms is not None:
        cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (x_pos, y_pos), font, font_scale, color_text, thickness)

    # Draw risk bar at bottom
    bar_height = 20
    bar_y = h - bar_height - 10
    bar_width = int(w * 0.8)
    bar_x = int(w * 0.1)

    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

    # Risk level bar
    risk_bar_width = int(bar_width * risk)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + risk_bar_width, bar_y + bar_height), risk_color, -1)

    # Threshold markers
    avoid_thresh_x = int(bar_x + bar_width * 0.55)
    stop_thresh_x = int(bar_x + bar_width * 0.8)
    cv2.line(frame, (avoid_thresh_x, bar_y), (avoid_thresh_x, bar_y + bar_height), (255, 165, 0), 2)
    cv2.line(frame, (stop_thresh_x, bar_y), (stop_thresh_x, bar_y + bar_height), (255, 0, 0), 2)

    return frame
