"""Tests for choosing where the work runs.

The rule is short - a busy GPU means the CPU - and the tests are mostly about
the two ways it can be wrong: claiming the GPU is free when nobody could tell,
and falling back to the CPU for a reason nobody can read. Both produce a run
that is quietly ten times slower.
"""

import os
import sys
import unittest
from unittest import mock

import resources


class FakeCuda:
    """Stands in for torch.cuda without a GPU being present."""

    def __init__(self, available=True, free=1, total=4, report_error=None):
        self.available = available
        self._free = free
        self._total = total
        self._report_error = report_error

    def is_available(self):
        return self.available

    def synchronize(self, device=None):
        return None

    def is_bf16_supported(self):
        return False

    def get_device_name(self, index):
        return "fake"

    def mem_get_info(self, index):
        if self._report_error is not None:
            raise self._report_error
        return self._free, self._total


def fake_torch(cuda):
    """A module-shaped object with just the surface resources.py touches."""

    class Torch:
        pass

    module = Torch()
    module.cuda = cuda
    return module


class GpuBusyTests(unittest.TestCase):
    def test_an_absent_gpu_is_not_busy_because_there_is_nothing_to_take(self):
        cuda = FakeCuda(available=False)
        with mock.patch.dict(sys.modules, {"torch": fake_torch(cuda)}):
            device, reason = resources.choose_device()
        self.assertEqual(device, "cpu")
        self.assertIn("no CUDA device", reason)

    def test_a_mostly_free_gpu_is_not_busy(self):
        # 1 GB free of 4 GB is a quarter used, comfortably under the threshold.
        cuda = FakeCuda(available=True, free=3, total=4)
        with mock.patch.dict(sys.modules, {"torch": fake_torch(cuda)}):
            busy, note = resources.gpu_is_busy()
        self.assertFalse(busy)
        self.assertIn("25% of GPU memory in use", note)

    def test_a_busy_gpu_is_reported_with_the_number_that_decided_it(self):
        cuda = FakeCuda(available=True, free=1, total=10)
        with mock.patch.dict(sys.modules, {"torch": fake_torch(cuda)}):
            busy, note = resources.gpu_is_busy()
        self.assertTrue(busy)
        self.assertIn("90% of", note)

    def test_a_driver_that_will_not_report_is_treated_as_busy(self):
        # The conservative direction. Guessing free and fighting somebody else's
        # job for the card is worse than guessing busy and running slower.
        cuda = FakeCuda(available=True, report_error=RuntimeError("driver busy"))
        with mock.patch.dict(sys.modules, {"torch": fake_torch(cuda)}):
            busy, note = resources.gpu_is_busy()
        self.assertTrue(busy)
        self.assertIn("would not report", note)


class ChooseDeviceTests(unittest.TestCase):
    def test_a_free_gpu_is_used(self):
        cuda = FakeCuda(available=True, free=4, total=10)
        with mock.patch.dict(sys.modules, {"torch": fake_torch(cuda)}):
            device, reason = resources.choose_device()
        self.assertEqual(device, "cuda")
        self.assertIn("GPU", reason)

    def test_a_busy_gpu_gives_the_cpu_and_says_why(self):
        cuda = FakeCuda(available=True, free=1, total=10)
        with mock.patch.dict(sys.modules, {"torch": fake_torch(cuda)}):
            device, reason = resources.choose_device()
        self.assertEqual(device, "cpu")
        self.assertIn("left to whatever is already using it", reason)

    def test_the_threshold_is_configurable(self):
        # "Busy" is not one number: a machine running a job at ninety per cent
        # wants the card left alone, and one at half does not care.
        cuda = FakeCuda(available=True, free=5, total=10)
        with mock.patch.dict(sys.modules, {"torch": fake_torch(cuda)}):
            self.assertEqual(resources.choose_device(threshold=0.80)[0], "cuda")
            self.assertEqual(resources.choose_device(threshold=0.40)[0], "cpu")


class HostMemoryTests(unittest.TestCase):
    def test_host_memory_is_reported_rather_than_acted_on(self):
            # Host RAM is not a device tensor work can run on. It is reported
            # because it is half of the picture, and a run that fell "back" to it
            # would be describing something that does not exist. Only two devices
            # are ever returned.
            share = resources.host_memory_share()
            if share is not None:
                self.assertGreaterEqual(share, 0.0)
                self.assertLessEqual(share, 1.0)
            with mock.patch.dict(sys.modules, {"torch": fake_torch(FakeCuda())}):
                self.assertEqual(resources.choose_device()[0], "cuda")

    def test_describe_reports_everything_a_person_would_ask(self):
        report = resources.describe()
        for key in ("device", "reason", "gpu_busy", "gpu_note", "host_memory_share"):
            self.assertIn(key, report)


class ConfigWiringTests(unittest.TestCase):
    def test_the_device_is_chosen_from_the_machine_when_not_set(self):
        # A default of "cuda" whenever a GPU exists puts a long unattended run
        # on a card somebody else may be using, so the default has to be a
        # decision rather than an assumption.
        from config import Config, load_config

        self.assertEqual(Config().device, "", "the default must defer to a decision")
        config = load_config()
        self.assertIn(config.device, ("cuda", "cpu"))
        self.assertTrue(config.device_reason, "the choice must be explained")

    def test_an_explicit_device_is_honoured_and_attributed(self):
        previous = os.environ.get("JUPITER_DEVICE")
        os.environ["JUPITER_DEVICE"] = "cpu"
        try:
            from config import load_config

            config = load_config()
            self.assertEqual(config.device, "cpu")
            self.assertIn("JUPITER_DEVICE", config.device_reason)
        finally:
            if previous is None:
                os.environ.pop("JUPITER_DEVICE", None)
            else:
                os.environ["JUPITER_DEVICE"] = previous


class DependencyTests(unittest.TestCase):
    def test_resources_is_reachable_from_the_entry_point(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parents[1] / "run_jupiter.bat").read_text(
            encoding="utf-8"
        )
        self.assertIn("resources.py", text)
        self.assertIn('if /i "%~1"=="resources"', text)


if __name__ == "__main__":
    unittest.main()
