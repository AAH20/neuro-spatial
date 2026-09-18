"""
Sub-Millisecond Obstacle & Target Tracker.
Tracks moving obstacles directly from event streams with continuous ODE velocity updates.
"""
from typing import List, Tuple, Dict, Any
from ..event_stream.event_buffer import NeuromorphicEvent
from ..neural_ode.rk4_integrator import ContinuousNeuralODEIntegrator

class SubMillisecondObjectTracker:
    def __init__(self):
        # State: [pos_x, pos_y, vel_x, vel_y]
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.last_timestamp_us = 0

    def process_event_batch(self, events: List[NeuromorphicEvent]) -> Dict[str, Any]:
        if not events:
            return {"tracking": False, "state": self.state}

        # Cluster centroid of incoming microsecond events
        mean_x = sum(e.x for e in events) / float(len(events))
        mean_y = sum(e.y for e in events) / float(len(events))
        current_time_us = events[-1].timestamp_us

        if self.last_timestamp_us > 0 and current_time_us > self.last_timestamp_us:
            dt_s = (current_time_us - self.last_timestamp_us) / 1_000_000.0

            # Kinematic derivative ODE: dx/dt = vx, dv/dt = spring-damper to centroid
            def ode_fn(t: float, s: List[float]) -> List[float]:
                px, py, vx, vy = s
                kp = 50.0  # proportional convergence gain
                kd = 10.0  # velocity damping
                ax = kp * (mean_x - px) - kd * vx
                ay = kp * (mean_y - py) - kd * vy
                return [vx, vy, ax, ay]

            self.state = ContinuousNeuralODEIntegrator.rk4_step(ode_fn, 0.0, self.state, dt_s)
        else:
            self.state[0] = mean_x
            self.state[1] = mean_y

        self.last_timestamp_us = current_time_us

        return {
            "tracking": True,
            "events_processed": len(events),
            "estimated_target_x": round(self.state[0], 2),
            "estimated_target_y": round(self.state[1], 2),
            "estimated_vel_x": round(self.state[2], 2),
            "estimated_vel_y": round(self.state[3], 2),
            "latency_us": round((events[-1].timestamp_us - events[0].timestamp_us), 1)
        }
