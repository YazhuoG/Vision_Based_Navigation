"""
Stateful buffers and hysteresis logic for smoothing navigation signals.

Provides temporal smoothing, hysteresis thresholds, and oscillation detection.
"""

from typing import Dict, List, Optional
from collections import deque
import numpy as np


class RollingBuffer:
    """
    Fixed-size rolling buffer for temporal smoothing.
    """

    def __init__(self, window_size: int):
        """
        Args:
            window_size: Maximum number of elements to store
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def push(self, value: float) -> None:
        """Add a new value to the buffer."""
        self.buffer.append(value)

    def mean(self) -> float:
        """Compute mean of buffered values."""
        if len(self.buffer) == 0:
            return 0.0
        return float(np.mean(self.buffer))

    def median(self) -> float:
        """Compute median of buffered values."""
        if len(self.buffer) == 0:
            return 0.0
        return float(np.median(self.buffer))

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) == self.window_size

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class HysteresisThreshold:
    """
    Hysteresis thresholding to prevent rapid state oscillations.

    Uses separate thresholds for entering and leaving a state.
    """

    def __init__(
        self,
        enter_threshold: float,
        leave_threshold: float,
        initial_state: bool = False,
    ):
        """
        Args:
            enter_threshold: Value must exceed this to enter high state
            leave_threshold: Value must drop below this to leave high state
            initial_state: Initial state (True = high, False = low)
        """
        assert enter_threshold > leave_threshold, "Enter threshold must be > leave threshold"
        self.enter_threshold = enter_threshold
        self.leave_threshold = leave_threshold
        self.state = initial_state

    def update(self, value: float) -> bool:
        """
        Update state based on current value and return new state.

        Args:
            value: Current measurement

        Returns:
            state: True if in high state, False otherwise
        """
        if self.state:
            # Currently high, check if we should leave
            if value < self.leave_threshold:
                self.state = False
        else:
            # Currently low, check if we should enter
            if value > self.enter_threshold:
                self.state = True

        return self.state

    def reset(self, state: bool = False) -> None:
        """Reset to initial state."""
        self.state = state


class OscillationDetector:
    """
    Detects oscillating behavior (rapid flip-flopping between states).
    """

    def __init__(self, window_size: int, flip_threshold: int):
        """
        Args:
            window_size: Number of recent decisions to track
            flip_threshold: Number of flips within window to trigger oscillation
        """
        self.window_size = window_size
        self.flip_threshold = flip_threshold
        self.history = deque(maxlen=window_size)

    def push(self, direction: str) -> bool:
        """
        Add a new direction decision and check for oscillation.

        Args:
            direction: "left" or "right"

        Returns:
            oscillating: True if oscillation detected
        """
        self.history.append(direction)

        if len(self.history) < 3:
            return False

        # Count direction changes (flips)
        flips = 0
        for i in range(1, len(self.history)):
            if self.history[i] != self.history[i - 1]:
                flips += 1

        return flips >= self.flip_threshold

    def clear(self) -> None:
        """Clear history."""
        self.history.clear()


class NavigationMemory:
    """
    Manages temporal smoothing and state memory for navigation.
    """

    def __init__(self, config: Dict):
        """
        Initialize memory with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Rolling buffers for smoothing
        risk_window = config["risk"]["risk_smooth_window"]
        openness_window = config["risk"]["openness_smooth_window"]

        self.risk_buffer = RollingBuffer(risk_window)
        self.openness_buffer = RollingBuffer(openness_window)

        # Hysteresis thresholds for risk levels
        self.avoid_hysteresis = HysteresisThreshold(
            enter_threshold=config["risk"]["thresholds"]["avoid"],
            leave_threshold=config["risk"]["hysteresis"]["leave_avoid"],
            initial_state=False,
        )

        self.stop_hysteresis = HysteresisThreshold(
            enter_threshold=config["risk"]["thresholds"]["stop"],
            leave_threshold=config["risk"]["hysteresis"]["leave_stop"],
            initial_state=False,
        )

        # Oscillation detector
        self.oscillation_detector = OscillationDetector(
            window_size=config["memory"]["oscillation_window"],
            flip_threshold=config["memory"]["oscillation_flips"],
        )

        # Cooldown timer
        self.cooldown_frames = config["memory"]["cooldown_frames"]
        self.cooldown_counter = 0
        self.last_direction = None

    def update(
        self,
        risk_raw: float,
        openness_raw: float,
        preferred_direction: str,
    ) -> Dict:
        """
        Update memory with new observations and return smoothed signals.

        Args:
            risk_raw: Raw risk score [0, 1]
            openness_raw: Raw openness score
            preferred_direction: "left" or "right"

        Returns:
            signals: Dictionary with smoothed values and flags
        """
        # Update buffers
        self.risk_buffer.push(risk_raw)
        self.openness_buffer.push(openness_raw)

        # Compute smoothed values
        risk_smooth = self.risk_buffer.mean()
        openness_smooth = self.openness_buffer.mean()

        # Update hysteresis states
        avoid_state = self.avoid_hysteresis.update(risk_smooth)
        stop_state = self.stop_hysteresis.update(risk_smooth)

        # Update oscillation detector
        oscillating = self.oscillation_detector.push(preferred_direction)

        # Update cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            in_cooldown = True
        else:
            in_cooldown = False

        # If direction changed, start cooldown
        if preferred_direction != self.last_direction:
            self.cooldown_counter = self.cooldown_frames
            self.last_direction = preferred_direction

        return {
            "risk_raw": risk_raw,
            "risk_smooth": risk_smooth,
            "openness_raw": openness_raw,
            "openness_smooth": openness_smooth,
            "preferred_direction": preferred_direction,
            "avoid_state": avoid_state,
            "stop_state": stop_state,
            "oscillating": oscillating,
            "in_cooldown": in_cooldown,
        }

    def reset(self) -> None:
        """Reset all memory state."""
        self.risk_buffer.clear()
        self.openness_buffer.clear()
        self.avoid_hysteresis.reset()
        self.stop_hysteresis.reset()
        self.oscillation_detector.clear()
        self.cooldown_counter = 0
        self.last_direction = None
