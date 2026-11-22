"""A restart waits for the RAM an ended engine pinned: a start sized meanwhile plans a RAM tier too small for the model."""
import unittest
from contextlib import redirect_stdout
from io import StringIO

from serve.engine import process
from serve.engine.process import ProcessEngine

GIB = 1 << 30


class Clock:
    def __init__(self, readings):
        self.t, self.readings = 0.0, list(readings)

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def available(self):   # one reading per call; the last one stays
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]


def wait(before, readings, **kw):
    eng = ProcessEngine("basalt", ["--model", "m.basalt"], lazy=True)
    c = Clock(readings)
    out = StringIO()
    with redirect_stdout(out):
        ok = eng.await_release(before, sleep=c.sleep, now=c.now, available=c.available, **kw)
    return ok, c.t, out.getvalue()


class ReleaseWait(unittest.TestCase):
    def test_nothing_to_wait_for(self):
        ok, t, out = wait(110 * GIB, [108 * GIB])
        self.assertTrue(ok)
        self.assertEqual(t, 0.0)
        self.assertEqual(out, "")

    def test_waits_until_the_memory_is_back(self):
        # the dead engine's 78 GiB comes back over a few seconds: 53 GiB free at first, as in the 2026-11-22 restart
        ok, t, out = wait(110 * GIB, [53 * GIB, 53 * GIB, 60 * GIB, 80 * GIB, 100 * GIB, 108 * GIB])
        self.assertTrue(ok)
        self.assertAlmostEqual(t, 2.5)
        self.assertIn("waiting for the last engine's memory: 53.0 GiB", out)
        self.assertIn("memory back: 108.0 GiB", out)

    def test_a_reading_that_stopped_growing_ends_the_wait(self):
        ok, t, out = wait(110 * GIB, [53 * GIB, 70 * GIB, 90 * GIB])
        self.assertFalse(ok)
        self.assertLess(t, process.RELEASE_WAIT_S)
        self.assertGreaterEqual(t, process.RELEASE_STEADY_S)
        self.assertIn("another program holds the rest", out)

    def test_bounded(self):
        readings = [50 * GIB + i * (process.RELEASE_GROWTH + 1) for i in range(1000)]   # growing slowly forever
        ok, t, out = wait(500 * GIB, readings, wait_s=20.0)
        self.assertFalse(ok)
        self.assertLessEqual(t, 21.0)
        self.assertIn("not back after 20 s", out)

    def test_no_reading(self):
        ok, t, _ = wait(None, [53 * GIB])
        self.assertTrue(ok)
        ok, t, _ = wait(110 * GIB, [None])
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
