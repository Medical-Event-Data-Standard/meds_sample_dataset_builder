from pathlib import Path

from meds_testing_helpers.dataset import MEDSDataset
from meds_testing_helpers.static_sample_data import exported_yamls


def recursive_check(d: dict, full_key: str | None = None):
    for k, v in d.items():
        local_key = k if full_key is None else f"{full_key}/{k}"
        if isinstance(v, dict):
            recursive_check(v, full_key=local_key)
        else:
            try:
                MEDSDataset.from_yaml(v)
            except Exception as e:
                raise AssertionError(f"Failed to parse {local_key}") from e


def test_static_datasets():
    assert len(exported_yamls) > 0
    recursive_check(exported_yamls)


def test_fixture(simple_static_MEDS: Path):
    try:
        data = MEDSDataset(root_dir=simple_static_MEDS)
        assert data.task_labels is None
        data.subject_splits
        data.dataset_metadata
        data.code_metadata
        data.data_shards
    except Exception as e:
        raise AssertionError(f"Failed to load {simple_static_MEDS}") from e


def test_fixture_with_task(simple_static_MEDS_dataset_with_task: Path):
    try:
        data = MEDSDataset(root_dir=simple_static_MEDS_dataset_with_task)
        assert data.task_labels is not None
        data.subject_splits
        data.dataset_metadata
        data.code_metadata
        data.data_shards
    except Exception as e:
        raise AssertionError(f"Failed to load {simple_static_MEDS_dataset_with_task}") from e
