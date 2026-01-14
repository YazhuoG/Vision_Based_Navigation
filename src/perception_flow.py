"""
Optical flow based perception for obstacle risk and openness estimation.

Mode A: Uses dense optical flow to estimate navigation signals without ML.
"""

from typing import Dict, Tuple
import numpy as np
import cv2


def compute_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    config: Dict,
) -> np.ndarray:
    """
    Compute dense optical flow between two consecutive grayscale frames.

    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        config: Configuration dictionary

    Returns:
        flow: Optical flow array of shape (H, W, 2) with (dx, dy) at each pixel
    """
    method = config["flow"]["method"]

    if method == "farneback":
        params = config["flow"]["farneback"]
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=params["pyr_scale"],
            levels=params["levels"],
            winsize=params["winsize"],
            iterations=params["iterations"],
            poly_n=params["poly_n"],
            poly_sigma=params["poly_sigma"],
            flags=params["flags"],
        )
    elif method == "dis":
        preset = config["flow"]["dis"]["preset"]
        # DIS optical flow - faster alternative
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        if preset == "ultrafast":
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        elif preset == "fast":
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        flow = dis.calc(prev_gray, curr_gray, None)
    else:
        raise ValueError(f"Unknown optical flow method: {method}")

    return flow


def compute_flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of optical flow.

    Args:
        flow: Flow array (H, W, 2)

    Returns:
        magnitude: Flow magnitude (H, W)
    """
    return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)


def compute_obstacle_risk(
    flow: np.ndarray,
    config: Dict,
) -> float:
    """
    Estimate obstacle risk from flow magnitude in central region.

    High flow magnitude in the central region indicates potential obstacles
    in the forward path (camera motion causes high relative motion for close objects).

    Args:
        flow: Optical flow array (H, W, 2)
        config: Configuration dictionary

    Returns:
        risk_score: Normalized risk score [0, 1]
    """
    magnitude = compute_flow_magnitude(flow)
    h, w = magnitude.shape

    # Define central region (horizontal span)
    central_region = config["risk"]["central_region"]
    x_start = int(w * central_region[0])
    x_end = int(w * central_region[1])

    # Extract central region magnitude
    central_mag = magnitude[:, x_start:x_end]

    # Clip high values to avoid outliers
    mag_clip = config["risk"]["magnitude_clip"]
    central_mag_clipped = np.clip(central_mag, 0, mag_clip)

    # Compute mean magnitude in central region
    mean_mag = np.mean(central_mag_clipped)

    # Normalize to [0, 1]
    risk_score = mean_mag / mag_clip

    return float(risk_score)


def compute_openness(
    flow: np.ndarray,
    config: Dict,
) -> Tuple[float, str]:
    """
    Estimate which side (left/right) has more openness based on flow.

    Lower flow magnitude on one side suggests more open space (farther objects).
    Higher magnitude suggests closer obstacles.

    Args:
        flow: Optical flow array (H, W, 2)
        config: Configuration dictionary

    Returns:
        openness_score: Difference in flow magnitude (left - right), normalized
        preferred_direction: "left" or "right" indicating more open side
    """
    magnitude = compute_flow_magnitude(flow)
    h, w = magnitude.shape

    # Split left and right
    split_ratio = config["risk"]["left_right_split"]
    split_x = int(w * split_ratio)

    left_region = magnitude[:, :split_x]
    right_region = magnitude[:, split_x:]

    # Clip magnitudes
    mag_clip = config["risk"]["magnitude_clip"]
    left_mag = np.clip(left_region, 0, mag_clip)
    right_mag = np.clip(right_region, 0, mag_clip)

    # Compute mean magnitudes
    left_mean = np.mean(left_mag)
    right_mean = np.mean(right_mag)

    # Openness score: positive means right is more open, negative means left is more open
    openness_score = (left_mean - right_mean) / mag_clip

    # Determine preferred direction (turn toward more open side)
    preferred_direction = "right" if openness_score > 0 else "left"

    return float(openness_score), preferred_direction


def visualize_flow(
    frame: np.ndarray,
    flow: np.ndarray,
    step: int = 16,
) -> np.ndarray:
    """
    Visualize optical flow as arrows overlaid on frame.

    Args:
        frame: RGB frame
        flow: Optical flow array (H, W, 2)
        step: Sampling step for arrows

    Returns:
        vis_frame: Frame with flow vectors drawn
    """
    vis_frame = frame.copy()
    h, w = flow.shape[:2]

    # Sample flow vectors
    y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)

    for y, x in zip(y_coords, x_coords):
        fx, fy = flow[y, x]
        # Draw arrow
        cv2.arrowedLine(
            vis_frame,
            (x, y),
            (int(x + fx), int(y + fy)),
            (0, 255, 0),
            1,
            tipLength=0.3,
        )

    return vis_frame


def visualize_flow_hsv(flow: np.ndarray) -> np.ndarray:
    """
    Visualize optical flow using HSV color coding.
    Hue = direction, Saturation = magnitude.

    Args:
        flow: Optical flow array (H, W, 2)

    Returns:
        flow_rgb: RGB visualization of flow
    """
    h, w = flow.shape[:2]

    # Compute flow in polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255  # Saturation: full
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude

    # Convert to RGB
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return flow_rgb
