"""Pytest plugin capabilities for loading sample MEDS datasets."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from .dataset import MEDSDataset
from .static_sample_data import (
    SIMPLE_STATIC_SHARDED_BY_SPLIT,
    SIMPLE_STATIC_SHARDED_BY_SPLIT_WITH_TASKS,
)


def pytest_addoption(parser):  # pragma: no cover
    parser.addoption(
        "--generated-dataset-N",
        type=int,
        help="If using the generated-dataset dataset, how many patients?",
        default=500,
    )

    parser.addoption(
        "--generated-dataset-seed",
        type=int,
        help="If generating a dataset, what seed to use?",
        default=1,
    )

    parser.addoption(
        "--MEDS-datasets-scope",
        type=str,
        help="For generated or static dataset fixtures, what scope to use?",
        default="session",
    )


def get_MEDS_datasets_scope(fixture_name: str, config) -> str:
    return config.getoption("--MEDS-datasets-scope")


@pytest.fixture(scope=get_MEDS_datasets_scope)
def simple_static_MEDS() -> Path:
    with tempfile.TemporaryDirectory() as data_dir:
        data_dir = Path(data_dir)
        data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)
        data.write(data_dir)
        yield data_dir


@pytest.fixture(scope=get_MEDS_datasets_scope)
def simple_static_MEDS_dataset_with_task() -> Path:
    with tempfile.TemporaryDirectory() as data_dir:
        data_dir = Path(data_dir)
        data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT_WITH_TASKS)
        data.write(data_dir)
        yield data_dir


def generate_MEDS(request, dataset_spec: str) -> Path:
    N = request.config.getoption("--generated-dataset-N")
    seed = request.config.getoption("--generated-dataset-seed")
    with tempfile.TemporaryDirectory() as data_dir:
        data_dir = Path(data_dir)

        cmd_args = [
            "build_sample_MEDS_dataset",
            f"seed={seed}",
            f"N_subjects={N}",
            "do_overwrite=False",
            f"output_dir={data_dir!s}",
            "dataset_spec/data_generator=sample",
        ]

        out = subprocess.run(cmd_args, shell=False, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        if out.returncode != 0:  # pragma: no cover
            raise RuntimeError(error_str)

        yield data_dir


@pytest.fixture(scope=get_MEDS_datasets_scope)
def generated_sample_MEDS(request) -> Path:
    yield from generate_MEDS(request, "sample")
