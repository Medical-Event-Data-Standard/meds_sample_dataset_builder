import subprocess
import tempfile
from pathlib import Path

from meds_testing_helpers.dataset import MEDSDataset


def test_dataset_generation(generated_sample_MEDS):
    """Test the generation of a dataset."""

    # This will throw an error if the generated data is invalid
    MEDSDataset(root_dir=generated_sample_MEDS)


def test_error_cases():
    cmd_args = [
        "build_sample_MEDS_dataset",
        "seed=1",
        "N_subjects=10",
        "dataset_spec/data_generator=sample",
    ]
    with tempfile.TemporaryDirectory() as data_dir:
        output_is_file = Path(data_dir) / "output_is_file"
        output_is_file.touch()

        out = subprocess.run(
            [*cmd_args, f"output_dir={output_is_file!s}"],
            shell=False,
            check=False,
            capture_output=True,
        )
        assert out.returncode != 0, "Should fail because output is a file"

        output_is_existing = Path(data_dir) / "output_is_existing"
        out_data = output_is_existing / "data"

        out_data.mkdir(parents=True)

        (out_data / "data1.parquet").touch()
        out = subprocess.run(
            [*cmd_args, f"output_dir={output_is_existing!s}"],
            shell=False,
            check=False,
            capture_output=True,
        )
        assert out.returncode != 0, "Should fail because output data subdir exists"

        out = subprocess.run(
            [*cmd_args, f"output_dir={output_is_existing!s}", "do_overwrite=True"],
            shell=False,
            check=False,
            capture_output=True,
        )
        assert out.returncode == 0, "Should succeed because output data subdir exists but overwrite is True"
