import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from continuumbench_experiments.models.adapters import OfficialRelationalTransformerAdapter


class OfficialRtAdapterTests(unittest.TestCase):
    def test_official_rt_rejects_non_linux_runtime_early(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "rt").mkdir()
            (repo_path / "rt" / "main.py").write_text("def main(**kwargs):\n    return None\n")

            adapter = OfficialRelationalTransformerAdapter(
                dataset_name="rel-f1",
                task_name="driver-top3",
                target_col="qualifying",
                rt_repo_path=str(repo_path),
                python_executable="python",
            )

            with mock.patch(
                "continuumbench_experiments.models.adapters.platform.system",
                return_value="Darwin",
            ):
                with self.assertRaisesRegex(RuntimeError, "Linux \\+ CUDA"):
                    adapter.fit(train_data=None, val_data=None, task=None)

    def test_official_rt_rejects_python_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "rt").mkdir()
            (repo_path / "rt" / "main.py").write_text("def main(**kwargs):\n    return None\n")

            adapter = OfficialRelationalTransformerAdapter(
                dataset_name="rel-f1",
                task_name="driver-top3",
                target_col="qualifying",
                rt_repo_path=str(repo_path),
                python_executable="python",
            )

            probe = subprocess.CompletedProcess(
                args=["python"],
                returncode=0,
                stdout='{"platform":"Linux","cuda_available":false}\n',
                stderr="",
            )

            with mock.patch(
                "continuumbench_experiments.models.adapters.platform.system",
                return_value="Linux",
            ):
                with mock.patch(
                    "continuumbench_experiments.models.adapters.subprocess.run",
                    return_value=probe,
                ):
                    with self.assertRaisesRegex(RuntimeError, "cuda_available=False"):
                        adapter.fit(train_data=None, val_data=None, task=None)


if __name__ == "__main__":
    unittest.main()
