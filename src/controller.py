"""
Mapping from FSM outputs to simulated robot motion commands.
"""

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class MotionCommand:
    """
    Simulated robot motion command.
    """
    linear_velocity: float  # Forward velocity (m/s)
    angular_velocity: float  # Turning rate (rad/s, positive = left)
    command_name: str  # Human-readable command name

    def __str__(self) -> str:
        return f"{self.command_name}: v={self.linear_velocity:.2f}, w={self.angular_velocity:.2f}"


class NavigationController:
    """
    Converts FSM state to motion commands.
    """

    def __init__(self, config: Dict):
        """
        Initialize controller with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.linear_v = config["controller"]["linear_velocity"]
        self.angular_w = config["controller"]["angular_velocity"]
        self.search_w = config["controller"]["search_turn"]

    def get_command(self, state: str, preferred_direction: str = None) -> MotionCommand:
        """
        Generate motion command based on current FSM state.

        Args:
            state: Current FSM state name
            preferred_direction: "left" or "right" for directional states

        Returns:
            command: Motion command
        """
        if state == "CRUISE_FORWARD":
            return MotionCommand(
                linear_velocity=self.linear_v,
                angular_velocity=0.0,
                command_name="FORWARD",
            )

        elif state == "AVOID_LEFT":
            return MotionCommand(
                linear_velocity=self.linear_v * 0.5,  # Slow down while avoiding
                angular_velocity=self.angular_w,  # Turn left (positive)
                command_name="TURN_LEFT",
            )

        elif state == "AVOID_RIGHT":
            return MotionCommand(
                linear_velocity=self.linear_v * 0.5,
                angular_velocity=-self.angular_w,  # Turn right (negative)
                command_name="TURN_RIGHT",
            )

        elif state == "STOP_TOO_CLOSE":
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                command_name="STOP",
            )

        elif state == "RECOVERY":
            # Recovery: commit to one direction based on last known preference
            if preferred_direction == "left":
                return MotionCommand(
                    linear_velocity=self.linear_v * 0.3,
                    angular_velocity=self.angular_w,
                    command_name="RECOVERY_LEFT",
                )
            else:
                return MotionCommand(
                    linear_velocity=self.linear_v * 0.3,
                    angular_velocity=-self.angular_w,
                    command_name="RECOVERY_RIGHT",
                )

        elif state == "SEARCH_LEFT":
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=self.search_w,
                command_name="SEARCH_LEFT",
            )

        elif state == "SEARCH_RIGHT":
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=-self.search_w,
                command_name="SEARCH_RIGHT",
            )

        else:
            # Default: stop
            return MotionCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                command_name="STOP",
            )
