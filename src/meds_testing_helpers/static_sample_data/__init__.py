from importlib.resources import files

from .. import __package_name__

root = files(__package_name__).joinpath("static_sample_data")

static_dataset_yamls = root.rglob("*.yaml")

exported_yamls = {}
for yaml_path in static_dataset_yamls:
    yaml_name = yaml_path.relative_to(root).with_suffix("").as_posix()

    nested = [x.upper() for x in yaml_name.split("/")]
    current = exported_yamls
    for nested_key in nested[:-1]:
        current = current.setdefault(nested_key, {})
    current[nested[-1]] = yaml_path.read_text()

for n, val in exported_yamls.items():
    globals()[n] = val

__all__ = [*list(exported_yamls.keys()), "exported_yamls"]
