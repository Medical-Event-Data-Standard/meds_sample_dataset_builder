from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

__package_name__ = "meds_testing_helpers"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

GEN_YAML = files(__package_name__).joinpath("configs/generate_dataset.yaml")
INF_YAML = files(__package_name__).joinpath("configs/infer_dataset_config.yaml")

__all__ = [
    "GEN_YAML",
    "INF_YAML",
    "__package_name__",
    "__version__",
]
