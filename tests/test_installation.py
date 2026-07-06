# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Copyright 2025 Shad Nygren, Virtual Hipster Corporation
# Contributed to the OpenDDE project under the Apache License 2.0

"""
Test suite for installation and dependency compatibility issues.
"""

import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path


class TestInstallation(unittest.TestCase):
    """Test that package dependencies are compatible"""

    def test_python_version(self):
        """Verify Python version is compatible (3.11+)"""
        version_info = sys.version_info
        self.assertGreaterEqual(version_info.major, 3)
        self.assertGreaterEqual(
            version_info.minor, 11, "OpenDDE requires Python 3.11 or higher"
        )

    def test_required_packages_importable(self):
        """Verify core packages can be imported"""
        required_packages = [
            "torch",
            "numpy",
            "scipy",
            "pandas",
            "ml_collections",
        ]

        for package in required_packages:
            spec = importlib.util.find_spec(package)
            if spec is None:
                self.skipTest(
                    f"{package} not installed - this test documents missing dependencies"
                )

    def test_opendde_imports(self):
        """Test that opendde package can be imported when installed"""
        try:
            import opendde  # noqa: F401

            self.assertTrue(True)
        except ImportError:
            self.skipTest(
                "OpenDDE not installed - this documents the need for installation"
            )

    def test_cuda_availability(self):
        """Document CUDA availability for GPU acceleration"""
        try:
            import torch

            if torch.cuda.is_available():
                self.assertTrue(True, "CUDA is available")
            else:
                # Not a failure, just documentation
                print("Note: CUDA not available, will use CPU")
        except ImportError:
            self.skipTest("PyTorch not installed")

    def test_cpu_and_gpu_install_extras_are_declared(self):
        """Verify release install extras are exposed in package metadata."""
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())
        optional_dependencies = pyproject["project"]["optional-dependencies"]
        runtime_dependencies = pyproject["project"]["dependencies"]

        self.assertIn("cpu", optional_dependencies)
        self.assertIn("gpu", optional_dependencies)
        self.assertTrue(
            any("cuequivariance-torch" in dep for dep in optional_dependencies["gpu"])
        )
        self.assertTrue(
            any(
                "cuequivariance-ops-torch-cu12" in dep
                for dep in optional_dependencies["gpu"]
            )
        )
        self.assertFalse(any("cuequivariance" in dep for dep in runtime_dependencies))
        self.assertFalse(any("icecream" in dep for dep in runtime_dependencies))
        self.assertFalse(any("ipdb" in dep for dep in runtime_dependencies))

    def test_doctor_command_is_registered(self):
        """The public CLI should expose environment diagnostics."""
        from click.testing import CliRunner
        from runner.batch_inference import opendde_cli

        result = CliRunner().invoke(opendde_cli, ["doctor"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("OpenDDE environment", result.output)
        self.assertIn("pip install 'opendde[gpu]'", result.output)


if __name__ == "__main__":
    unittest.main()  # Test signed commit
