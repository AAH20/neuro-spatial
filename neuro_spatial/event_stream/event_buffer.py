"""
Microsecond Asynchronous Neuromorphic Event Buffer.
Ingests individual pixel polarity spikes (x, y, timestamp_us, polarity) at 1,000,000+ events/sec.
"""
from dataclasses import dataclass
from typing import List, Optional
import collections

@dataclass(frozen=True)
class NeuromorphicEvent:
    x: int             # Pixel X coordinate (0-1279 for HD DVS)
    y: int             # Pixel Y coordinate (0-719 for HD DVS)
    timestamp_us: int  # Microsecond timestamp
    polarity: int      # +1 (brightness increased) or -1 (brightness decreased)

class EventStreamBuffer:
    def __init__(self, capacity: int = 500_000):
        self._events = collections.deque(maxlen=capacity)

    def append(self, event: NeuromorphicEvent) -> None:
        self._events.append(event)

    def append_batch(self, events: List[NeuromorphicEvent]) -> None:
        self._events.extend(events)

    def get_time_slice(self, start_us: int, end_us: int) -> List[NeuromorphicEvent]:
        return [e for e in self._events if start_us <= e.timestamp_us <= end_us]

    def __len__(self) -> int:
        return len(self._events)
