"""
Finite-state machine for navigation command selection.

Implements anti-oscillation logic with cooldown and recovery behaviors.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class FSMState:
    """
    FSM state representation.
    """
    name: str
    entry_frame: int = 0  # Frame when state was entered


class NavigationFSM:
    """
    Finite State Machine for navigation control.

    States:
    - CRUISE_FORWARD: Normal forward motion
    - AVOID_LEFT: Avoiding obstacle by turning left
    - AVOID_RIGHT: Avoiding obstacle by turning right
    - STOP_TOO_CLOSE: Stopped due to close obstacle
    - RECOVERY: Anti-oscillation recovery mode (commits to one direction)
    - SEARCH_LEFT/RIGHT: Optional search mode
    """

    def __init__(self, config: Dict):
        """
        Initialize FSM with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.default_state = config["fsm"]["default_state"]
        self.recovery_state = config["fsm"]["recovery_state"]
        self.search_enabled = config["fsm"].get("search_enabled", False)

        # Recovery timer
        self.recovery_frames = config["memory"]["recovery_frames"]
        self.recovery_counter = 0
        self.recovery_direction = None

        # Current state
        self.current_state = FSMState(name=self.default_state)
        self.frame_count = 0

    def update(self, signals: Dict) -> str:
        """
        Update FSM state based on perception and memory signals.

        Args:
            signals: Dictionary from NavigationMemory.update()

        Returns:
            new_state: New state name
        """
        self.frame_count += 1

        # Extract signals
        risk_smooth = signals["risk_smooth"]
        openness_smooth = signals["openness_smooth"]
        preferred_direction = signals["preferred_direction"]
        avoid_state = signals["avoid_state"]
        stop_state = signals["stop_state"]
        oscillating = signals["oscillating"]

        # Check if in recovery mode
        if self.current_state.name == "RECOVERY":
            self.recovery_counter -= 1
            if self.recovery_counter <= 0:
                # Recovery complete, return to normal operation
                self._transition_to(self.default_state)
            else:
                # Stay in recovery
                return self.current_state.name

        # Handle oscillation detection - enter recovery
        if oscillating and self.current_state.name != "RECOVERY":
            self._enter_recovery(preferred_direction)
            return self.current_state.name

        # FSM transitions based on risk and openness
        if stop_state:
            # Too close - must stop
            self._transition_to("STOP_TOO_CLOSE")

        elif avoid_state:
            # Need to avoid - choose direction
            if preferred_direction == "left":
                self._transition_to("AVOID_LEFT")
            else:
                self._transition_to("AVOID_RIGHT")

        else:
            # Safe to cruise forward
            self._transition_to("CRUISE_FORWARD")

        return self.current_state.name

    def _transition_to(self, new_state: str) -> None:
        """
        Transition to a new state.

        Args:
            new_state: Target state name
        """
        if new_state != self.current_state.name:
            self.current_state = FSMState(name=new_state, entry_frame=self.frame_count)

    def _enter_recovery(self, direction: str) -> None:
        """
        Enter recovery mode to prevent oscillation.

        Args:
            direction: Direction to commit to during recovery
        """
        self.recovery_direction = direction
        self.recovery_counter = self.recovery_frames
        self._transition_to("RECOVERY")

    def get_recovery_direction(self) -> Optional[str]:
        """
        Get the direction committed during recovery.

        Returns:
            direction: "left" or "right", or None if not in recovery
        """
        if self.current_state.name == "RECOVERY":
            return self.recovery_direction
        return None

    def get_state_duration(self) -> int:
        """
        Get number of frames spent in current state.

        Returns:
            duration: Frame count since entering current state
        """
        return self.frame_count - self.current_state.entry_frame

    def reset(self) -> None:
        """Reset FSM to initial state."""
        self.current_state = FSMState(name=self.default_state)
        self.frame_count = 0
        self.recovery_counter = 0
        self.recovery_direction = None
