"""Pytest plugin capabilities for loading sample MEDS datasets."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from .dataset import MEDSDataset
from .static_sample_data import SIMPLE_STATIC_SHARDED_BY_SPLIT


def pytest_addoption(parser):
    parser.addoption(
        "--mimic-like-N", type=int, help="If using the mimic-like dataset, how many patients?", default=500
    )

    parser.addoption(
        "--generated-dataset-seed",
        type=int,
        help="If generating a dataset, what seed to use?",
        default=1,
    )


@pytest.fixture
def simple_static_MEDS() -> Path:
    with tempfile.TemporaryDirectory() as data_dir:
        data_dir = Path(data_dir)
        data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)
        data.write(data_dir)
        yield data_dir


@pytest.fixture
def generated_mimic_like_MEDS(request) -> Path:
    N = request.config.getoption("--mimic-like-N")
    seed = request.config.getoption("--generated-dataset-seed")
    with tempfile.TemporaryDirectory() as data_dir:
        data_dir = Path(data_dir)

        cmd_args = [
            "build_sample_MEDS_dataset",
            f"seed={seed}",
            f"N_subjects={N}",
            "do_overwrite=False",
            f"output_dir={str(data_dir)}",
            "dataset_spec/data_generator=mimic",
        ]

        out = subprocess.run(cmd_args, shell=False, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        if out.returncode != 0:
            raise RuntimeError(error_str)

        yield data_dir
