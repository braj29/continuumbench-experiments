import unittest

from continuumbench_experiments.models.tabular import (
    resolve_tabicl_device,
    resolve_tabpfn_device,
)


class DeviceResolutionTests(unittest.TestCase):
    def test_tabicl_auto_uses_cpu_on_macos(self):
        self.assertEqual(resolve_tabicl_device("auto", platform_name="Darwin"), "cpu")

    def test_tabpfn_auto_uses_cpu_on_macos(self):
        self.assertEqual(resolve_tabpfn_device("auto", platform_name="Darwin"), "cpu")

    def test_tabpfn_auto_stays_auto_off_macos(self):
        self.assertEqual(resolve_tabpfn_device("auto", platform_name="Linux"), "auto")


if __name__ == "__main__":
    unittest.main()
