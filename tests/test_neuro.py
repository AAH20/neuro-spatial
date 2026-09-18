import unittest
from neuro_spatial.event_stream.event_buffer import NeuromorphicEvent, EventStreamBuffer
from neuro_spatial.event_stream.spatio_temporal_surface import SpatioTemporalSurface
from neuro_spatial.neural_ode.rk4_integrator import ContinuousNeuralODEIntegrator
from neuro_spatial.perception.tracker import SubMillisecondObjectTracker

class TestNeuroSpatial(unittest.TestCase):
    def test_event_buffer_and_slice(self):
        buf = EventStreamBuffer(capacity=100)
        for i in range(50):
            buf.append(NeuromorphicEvent(x=i, y=i, timestamp_us=1000 + i * 100, polarity=1))
        self.assertEqual(len(buf), 50)
        sliced = buf.get_time_slice(1500, 2500)
        self.assertEqual(len(sliced), 11)

    def test_spatio_temporal_surface_decay(self):
        surface = SpatioTemporalSurface(width=100, height=100, decay_constant_us=1000.0)
        surface.update(NeuromorphicEvent(x=10, y=20, timestamp_us=5000, polarity=1))
        # Exact at event time: 1.0
        self.assertAlmostEqual(surface.get_intensity(10, 20, 5000), 1.0)
        # Decay after 1 tau: e^(-1) approx 0.3678
        self.assertAlmostEqual(surface.get_intensity(10, 20, 6000), 0.367879, places=4)

    def test_rk4_exponential_decay(self):
        # dy/dt = -y, exact solution y(t) = y0 * exp(-t)
        def f(t, h):
            return [-h[0]]
        h0 = [1.0]
        dt = 0.1
        h1 = ContinuousNeuralODEIntegrator.rk4_step(f, 0.0, h0, dt)
        import math
        self.assertAlmostEqual(h1[0], math.exp(-0.1), places=5)

    def test_sub_millisecond_tracker(self):
        tracker = SubMillisecondObjectTracker()
        events = [
            NeuromorphicEvent(x=100 + i, y=200, timestamp_us=1000 + i * 10, polarity=1)
            for i in range(10)
        ]
        res = tracker.process_event_batch(events)
        self.assertTrue(res["tracking"])
        self.assertEqual(res["events_processed"], 10)
        self.assertTrue(res["estimated_target_x"] > 0)

if __name__ == "__main__":
    unittest.main()
