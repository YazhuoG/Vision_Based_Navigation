"""
Metrics collection for FPS, latency, and state durations.
"""

from typing import Dict, List
import time
from collections import defaultdict, Counter
import numpy as np


class PerformanceMetrics:
    """
    Track FPS and latency metrics.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.frame_times = []
        self.latencies = []
        self.start_time = None
        self.last_frame_time = None

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.perf_counter()
        self.last_frame_time = self.start_time

    def record_frame(self, latency: float = None) -> None:
        """
        Record a frame processing event.

        Args:
            latency: Optional frame processing latency (seconds)
        """
        current_time = time.perf_counter()
        self.frame_times.append(current_time)

        if latency is not None:
            self.latencies.append(latency)

        self.last_frame_time = current_time

    def get_fps(self) -> float:
        """
        Compute average FPS.

        Returns:
            fps: Frames per second
        """
        if len(self.frame_times) < 2 or self.start_time is None:
            return 0.0

        elapsed = self.frame_times[-1] - self.start_time
        if elapsed <= 0:
            return 0.0

        return len(self.frame_times) / elapsed

    def get_latency_stats(self) -> Dict[str, float]:
        """
        Compute latency statistics.

        Returns:
            stats: Dictionary with mean, p50, p95, p99 latencies (milliseconds)
        """
        if len(self.latencies) == 0:
            return {
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        latencies_ms = np.array(self.latencies) * 1000.0  # Convert to ms

        return {
            "mean": float(np.mean(latencies_ms)),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "p99": float(np.percentile(latencies_ms, 99)),
        }

    def summary(self) -> Dict:
        """
        Get complete performance summary.

        Returns:
            summary: Dictionary with all metrics
        """
        latency_stats = self.get_latency_stats()

        return {
            "frames_processed": len(self.frame_times),
            "fps": self.get_fps(),
            "latency_ms": latency_stats,
        }


class StateMetrics:
    """
    Track state durations and transitions.
    """

    def __init__(self):
        """Initialize state metrics tracker."""
        self.state_history = []
        self.state_durations = defaultdict(int)
        self.transition_count = 0
        self.recovery_events = 0

    def record_state(self, state: str) -> None:
        """
        Record current state.

        Args:
            state: Current FSM state name
        """
        # Check for transition
        if len(self.state_history) > 0 and state != self.state_history[-1]:
            self.transition_count += 1

            # Count recovery events
            if state == "RECOVERY":
                self.recovery_events += 1

        self.state_history.append(state)
        self.state_durations[state] += 1

    def get_state_percentages(self) -> Dict[str, float]:
        """
        Get percentage of time spent in each state.

        Returns:
            percentages: Dictionary mapping state to percentage
        """
        total_frames = len(self.state_history)
        if total_frames == 0:
            return {}

        percentages = {}
        for state, count in self.state_durations.items():
            percentages[state] = 100.0 * count / total_frames

        return percentages

    def summary(self) -> Dict:
        """
        Get complete state metrics summary.

        Returns:
            summary: Dictionary with all state metrics
        """
        return {
            "total_frames": len(self.state_history),
            "state_percentages": self.get_state_percentages(),
            "state_transitions": self.transition_count,
            "recovery_events": self.recovery_events,
        }


class NavigationMetrics:
    """
    Combined metrics tracker for navigation performance.
    """

    def __init__(self):
        """Initialize combined metrics tracker."""
        self.perf = PerformanceMetrics()
        self.states = StateMetrics()

    def start(self) -> None:
        """Start metrics collection."""
        self.perf.start()

    def record(self, state: str, latency: float = None) -> None:
        """
        Record a frame.

        Args:
            state: Current FSM state
            latency: Frame processing latency (seconds)
        """
        self.perf.record_frame(latency)
        self.states.record_state(state)

    def summary(self) -> Dict:
        """
        Get complete navigation metrics summary.

        Returns:
            summary: Dictionary with all metrics
        """
        return {
            "performance": self.perf.summary(),
            "states": self.states.summary(),
        }

    def print_summary(self) -> None:
        """Print formatted metrics summary."""
        summary = self.summary()

        print("=" * 60)
        print("NAVIGATION METRICS SUMMARY")
        print("=" * 60)

        # Performance
        perf = summary["performance"]
        print(f"\nPerformance:")
        print(f"  Frames processed: {perf['frames_processed']}")
        print(f"  Average FPS: {perf['fps']:.2f}")

        latency = perf["latency_ms"]
        print(f"\n  Latency (ms):")
        print(f"    Mean:  {latency['mean']:.2f}")
        print(f"    P50:   {latency['p50']:.2f}")
        print(f"    P95:   {latency['p95']:.2f}")
        print(f"    P99:   {latency['p99']:.2f}")

        # States
        states = summary["states"]
        print(f"\nState Distribution:")
        for state, pct in sorted(states["state_percentages"].items(), key=lambda x: -x[1]):
            print(f"  {state:20s}: {pct:5.1f}%")

        print(f"\nState Transitions: {states['state_transitions']}")
        print(f"Recovery Events:   {states['recovery_events']}")
        print("=" * 60)
