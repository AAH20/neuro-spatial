"""
Time-Surface Exponential Decay Grid.
Transforms asynchronous point events into a continuous 2D surface of active edges
without artificial frame quantization.
"""
import math
from typing import Dict, Tuple, List
from .event_buffer import NeuromorphicEvent

class SpatioTemporalSurface:
    def __init__(self, width: int = 640, height: int = 480, decay_constant_us: float = 10000.0):
        self.width = width
        self.height = height
        self.decay_us = decay_constant_us
        self.surface: Dict[Tuple[int, int], int] = {}  # (x, y) -> latest timestamp_us

    def update(self, event: NeuromorphicEvent) -> None:
        if 0 <= event.x < self.width and 0 <= event.y < self.height:
            self.surface[(event.x, event.y)] = event.timestamp_us

    def get_intensity(self, x: int, y: int, query_time_us: int) -> float:
        last_t = self.surface.get((x, y))
        if last_t is None or query_time_us < last_t:
            return 0.0
        dt = query_time_us - last_t
        # Exponential time decay: I = exp(-dt / tau)
        return math.exp(-dt / self.decay_us)
