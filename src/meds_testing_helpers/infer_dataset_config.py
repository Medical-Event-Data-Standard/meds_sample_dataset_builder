#!/usr/bin/env python

import dataclasses
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from meds import (
    birth_code,
    code_field,
    dataset_metadata_filepath,
    death_code,
    held_out_split,
    subject_id_field,
    subject_splits_filepath,
    time_field,
    train_split,
    tuning_split,
)
from omegaconf import DictConfig, OmegaConf

from . import INF_YAML
from .dataset_generator import (
    DatetimeGenerator,
    MEDSDataDFGenerator,
    MEDSDatasetGenerator,
    PositiveIntGenerator,
    PositiveTimeDeltaGenerator,
    ProportionGenerator,
)

logger = logging.getLogger(__name__)


def to_dt_list(df: pl.DataFrame, col: str) -> list[str]:
    """Converts a Polars DataFrame column to a list of ISO format strings.

    Args:
        df: The DataFrame.
        col: The column name.

    Returns:
        A list of ISO format strings.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({"col": [datetime(2021, 1, 1), datetime(2021, 1, 2)]})
        >>> to_dt_list(df, "col")
        ['2021-01-01T00:00:00', '2021-01-02T00:00:00']
    """
    return df.select(pl.col(col).drop_nulls().dt.strftime("%Y-%m-%dT%H:%M:%S%.f"))[col].to_list()


@hydra.main(version_base=None, config_path=str(INF_YAML.parent), config_name=INF_YAML.stem)
def main(cfg: DictConfig):
    """Infers the configuration parameters that would generate a dataset similar to the input."""

    output_fp = Path(cfg.output_fp)
    if output_fp.exists():
        if cfg.do_overwrite:
            logger.info(f"Overwriting existing file {output_fp}.")
        else:
            raise FileExistsError(f"Output file {output_fp} already exists.")

    data_dir = Path(cfg.dataset_dir) / "data"
    metadata_dir = Path(cfg.dataset_dir) / "metadata"

    shards = list(data_dir.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {data_dir}")

    dataset_metadata_fp = metadata_dir / dataset_metadata_filepath
    if not dataset_metadata_fp.exists():
        logger.warning(f"Dataset metadata file not found at {dataset_metadata_fp}!")
        dataset_name = "UNKNOWN"
    else:
        dataset_metadata = json.loads(dataset_metadata_fp.read_text())
        dataset_name = f"{dataset_metadata['dataset_name']}/SYNTHETIC"

    subject_splits_fp = metadata_dir / subject_splits_filepath
    if not subject_splits_fp.exists():
        train_frac = None
        tuning_frac = None
        held_out_frac = None
    else:
        subject_splits = pl.read_parquet(subject_splits_fp)
        split_cnts = subject_splits.group_by("split").agg(pl.col(subject_id_field).count().alias("count"))
        train_cnt = split_cnts.filter(pl.col("split") == train_split).select("count").first().item()
        tuning_cnt = split_cnts.filter(pl.col("split") == tuning_split).select("count").first().item()
        held_out_cnt = split_cnts.filter(pl.col("split") == held_out_split).select("count").first().item()

        total_cnt = train_cnt + tuning_cnt + held_out_cnt
        train_frac = train_cnt / total_cnt
        tuning_frac = tuning_cnt / total_cnt
        held_out_frac = 1 - train_frac - tuning_frac

    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()

    rng.shuffle(shards)
    shards_to_examine = shards[:3]

    code_col = pl.col(code_field)
    time_col = pl.col(time_field)
    is_static = time_col.is_null()
    is_dynamic = time_col.is_not_null()
    is_birth = code_col.str.starts_with(birth_code)
    is_death = code_col.str.starts_with(death_code)
    is_dynamic_data = is_dynamic & ~is_birth & ~is_death
    numerics_present = (
        pl.col("numeric_value").is_not_null()
        & pl.col("numeric_value").is_finite()
        & pl.col("numeric_value").is_not_nan()
    )

    shards = [pl.read_parquet(shard, use_pyarrow=True) for shard in shards_to_examine]
    shard_sizes = [shard.select(pl.col(subject_id_field).n_unique()).item() for shard in shards]

    df = pl.concat(shards, how="vertical_relaxed")
    dynamic_df = df.filter(is_dynamic_data)

    static_vocab_size = df.filter(is_static).select(code_col.n_unique()).item()
    dynamic_vocab_size = dynamic_df.select(code_col.n_unique()).item()
    birth_codes_vocab_size = df.filter(is_birth).select(code_col.n_unique()).item()
    death_codes_vocab_size = df.filter(is_death).select(code_col.n_unique()).item()

    subject_stats = df.group_by(subject_id_field).agg(
        pl.when(is_birth).then(time_col).min().alias("birth_time"),
        pl.when(is_death).then(time_col).max().alias("death_time"),
        is_birth.any().alias("has_birth"),
        is_death.any().alias("has_death"),
        is_static.sum().alias("n_static_measurements"),
    )
    birth_times = to_dt_list(subject_stats, "birth_time")

    frac_subjects_with_birth = subject_stats.select(pl.col("has_birth").mean()).item()
    frac_subjects_with_death = subject_stats.select(pl.col("has_death").mean()).item()
    num_static_measurements_per_subject = subject_stats["n_static_measurements"].drop_nulls().to_list()

    subject_dynamic_stats = dynamic_df.group_by(subject_id_field).agg(
        time_col.min().alias("first_data_time"),
        time_col.max().alias("last_data_time"),
        time_col.n_unique().alias("n_events"),
        time_col.unique(maintain_order=True).diff().mean().alias("time_between_events"),
    )

    start_of_data_times = to_dt_list(subject_dynamic_stats, "first_data_time")
    num_events_per_subject = subject_dynamic_stats["n_events"].drop_nulls().to_list()
    time_between_data_events = subject_dynamic_stats["time_between_events"].drop_nulls().to_numpy()

    boundary_deltas = subject_dynamic_stats.join(subject_stats, on=subject_id_field).select(
        (pl.col("first_data_time") - pl.col("birth_time")).alias("time_between_birth_and_data"),
        (pl.col("death_time") - pl.col("last_data_time")).alias("time_between_data_and_death"),
    )

    time_between_birth_and_data = boundary_deltas["time_between_birth_and_data"].drop_nulls().to_numpy()
    time_between_data_and_death = boundary_deltas["time_between_data_and_death"].drop_nulls().to_numpy()

    num_measurements_per_event = (
        dynamic_df.group_by(subject_id_field, time_col).agg(pl.count())["count"].to_list()
    )

    dynamic_code_stats = (
        dynamic_df.group_by(code_field)
        .agg(numerics_present.sum().alias("n_values"), pl.count().alias("n_occurrences"))
        .select((pl.col("n_values") / pl.col("n_occurrences")).alias("frac_values"))
    )
    frac_dynamic_code_occurrences_with_values = dynamic_code_stats["frac_values"].to_list()

    static_code_stats = (
        df.filter(is_static)
        .group_by(code_field)
        .agg(numerics_present.sum().alias("n_values"), pl.count().alias("n_occurrences"))
        .select((pl.col("n_values") / pl.col("n_occurrences")).alias("frac_values"))
    )
    frac_static_code_occurrences_with_values = static_code_stats["frac_values"].to_list()

    birth_kwargs = {}
    if len(birth_times) > 0:
        birth_kwargs["birth_datetime_per_subject"] = DatetimeGenerator(birth_times)
        birth_kwargs["birth_codes_vocab_size"] = birth_codes_vocab_size
        birth_kwargs["time_between_birth_and_data_per_subject"] = PositiveTimeDeltaGenerator(
            time_between_birth_and_data
        )
    else:
        birth_kwargs["birth_datetime_per_subject"] = None
        birth_kwargs["birth_codes_vocab_size"] = 0
        birth_kwargs["time_between_birth_and_data_per_subject"] = None

    death_kwargs = {}
    if len(time_between_data_and_death) > 0:
        death_kwargs["death_codes_vocab_size"] = death_codes_vocab_size
        death_kwargs["time_between_data_and_death_per_subject"] = PositiveTimeDeltaGenerator(
            time_between_data_and_death
        )
    else:
        death_kwargs["death_codes_vocab_size"] = 0
        death_kwargs["time_between_data_and_death_per_subject"] = None

    data_generator = MEDSDataDFGenerator(
        start_data_datetime_per_subject=DatetimeGenerator(start_of_data_times),
        time_between_data_events_per_subject=PositiveTimeDeltaGenerator(time_between_data_events),
        num_events_per_subject=PositiveIntGenerator(num_events_per_subject),
        num_measurements_per_event=PositiveIntGenerator(num_measurements_per_event),
        num_static_measurements_per_subject=PositiveIntGenerator(num_static_measurements_per_subject),
        frac_dynamic_code_occurrences_with_value=ProportionGenerator(
            frac_dynamic_code_occurrences_with_values
        ),
        frac_static_code_occurrences_with_value=ProportionGenerator(frac_static_code_occurrences_with_values),
        static_vocab_size=static_vocab_size,
        dynamic_vocab_size=dynamic_vocab_size,
        frac_subjects_with_death=frac_subjects_with_death,
        frac_subjects_with_birth=frac_subjects_with_birth,
        **birth_kwargs,
        **death_kwargs,
    )

    dataset_generator = MEDSDatasetGenerator(
        data_generator=data_generator,
        shard_size=int(np.median(shard_sizes)),
        train_frac=train_frac,
        tuning_frac=tuning_frac,
        held_out_frac=held_out_frac,
        dataset_name=dataset_name,
    )

    dataset_config = DictConfig(dataclasses.asdict(dataset_generator))
    OmegaConf.save(dataset_config, output_fp)
