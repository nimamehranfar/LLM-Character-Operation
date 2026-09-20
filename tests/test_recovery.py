from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.training.recovery import RecoveryManager, atomic_torch_save, atomic_write_json, load_json


class RunRecoveryTests(unittest.TestCase):
    def test_atomic_json_and_torch_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            j = root / "state.json"
            p = root / "state.pt"
            atomic_write_json(j, {"x": 3})
            atomic_torch_save(p, {"tensor": torch.tensor([1, 2, 3])})
            self.assertEqual(load_json(j)["x"], 3)
            saved = torch.load(p, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(saved["tensor"], torch.tensor([1, 2, 3])))
            self.assertFalse(any(root.glob("*.tmp.*")))

    def test_completed_phase_marker_survives_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {"x": 1}
            manager = RecoveryManager.create_or_resume(
                checkpoint_root=root,
                config=cfg,
                config_path="config.toml",
                script_version="test",
                heartbeat_seconds=60,
                fresh=True,
                explicit_run_dir=None,
            )
            run_dir = manager.paths.run_dir
            manager.mark_stage_complete("phase_a_training", phase="phase_a")
            manager.close()

            resumed = RecoveryManager.create_or_resume(
                checkpoint_root=root,
                config=cfg,
                config_path="config.toml",
                script_version="test",
                heartbeat_seconds=60,
                fresh=False,
                explicit_run_dir=run_dir,
            )
            self.assertTrue(resumed.stage_complete("phase_a_training"))
            resumed.close()


    def test_active_elapsed_uses_durable_base_plus_current_session_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = RecoveryManager.create_or_resume(
                checkpoint_root=root,
                config={"x": 3},
                config_path="config.toml",
                script_version="test",
                heartbeat_seconds=60,
                fresh=True,
                explicit_run_dir=None,
            )
            manager._active_seconds_base = 12.5
            manager._session_perf_start = 100.0
            with patch("src.training.recovery.time.perf_counter", return_value=104.0):
                self.assertAlmostEqual(manager.active_elapsed_seconds(), 16.5, places=6)
                manager._save_state()
            self.assertAlmostEqual(load_json(manager.paths.state)["active_seconds"], 16.5, places=6)
            manager.close()

    def test_complete_run_is_not_auto_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {"x": 2}
            first = RecoveryManager.create_or_resume(
                checkpoint_root=root,
                config=cfg,
                config_path="config.toml",
                script_version="test",
                heartbeat_seconds=60,
                fresh=True,
                explicit_run_dir=None,
            )
            first_dir = first.paths.run_dir
            first.mark_complete()
            first.close()
            second = RecoveryManager.create_or_resume(
                checkpoint_root=root,
                config=cfg,
                config_path="config.toml",
                script_version="test",
                heartbeat_seconds=60,
                fresh=False,
                explicit_run_dir=None,
            )
            self.assertNotEqual(second.paths.run_dir, first_dir)
            second.close()


if __name__ == "__main__":
    unittest.main()
