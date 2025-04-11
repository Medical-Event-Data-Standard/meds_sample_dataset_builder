import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from meds import (
    DatasetMetadata,
    birth_code,
    code_field,
    death_code,
    description_field,
    held_out_split,
    numeric_value_field,
    parent_codes_field,
    subject_id_field,
    time_field,
    train_split,
    tuning_split,
)
from meds import __version__ as meds_version
from omegaconf import DictConfig

from . import GEN_YAML, __package_name__, __version__
from .dataset import MEDSDataset
from .rvs import (
    DatetimeGenerator,
    DiscreteGenerator,
    PositiveIntGenerator,
    PositiveTimeDeltaGenerator,
    ProportionGenerator,
)
from .types import (
    NON_NEGATIVE_INT,
    POSITIVE_INT,
    PROPORTION,
    is_NON_NEGATIVE_INT,
    is_POSITIVE_INT,
    is_PROPORTION,
)

logger = logging.getLogger(__name__)


@dataclass
class MEDSDataDFGenerator:
    """A class to generate whole dataset objects in the form of static and dynamic measurements.

    Attributes:
        birth_datetime_per_subject: A generator for the birth datetime of each subject.
        start_data_datetime_per_subject: A generator for the start datetime of data collection for each
            subject. Only used for subjects without a birth-date.
        time_between_birth_and_data_per_subject: A generator for the time between birth and data collection.
            Only used for subjects with a birth-date.
        time_between_data_and_death_per_subject: A generator for the time between data collection and death.
            Only used for subjects with a death-date.
        time_between_data_events_per_subject: A generator for the time between data events.
        num_events_per_subject: A generator for the number of events per subject.
        num_measurements_per_event: A generator for the number of measurements per event.
        num_static_measurements_per_subject: A generator for the number of static measurements per subject.
        frac_dynamic_code_occurrences_with_value: A generator for the proportion of dynamic codes with a
            numeric value.
        frac_static_code_occurrences_with_value: A generator for the proportion of static codes with a numeric
            value.
        static_vocab_size: The number of unique static codes.
        dynamic_vocab_size: The number of unique dynamic codes.
        frac_subjects_with_death: The proportion of subjects with a death date. The remaining subjects will
            have no death event in the data.
        frac_subjects_with_birth: The proportion of subjects with a birth date. The remaining subjects will
            have no birth event in the data.
        birth_codes_vocab_size: The number of unique birth codes. Birth codes will be of the form
            "{meds.birth_code}//{i}", or simply "{meds.birth_code}" if there is only one birth code.
        death_codes_vocab_size: The number of unique death codes. Death codes will be of the form
            "{meds.death_code}//{i}", or simply "{meds.death_code}" if there is only one death code.

    Raises:
        ValueError: Various validation errors for the input parameters will raise value errors.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> kwargs = dict(
        ...     birth_datetime_per_subject=DatetimeGenerator(
        ...         [np.datetime64("2001-01-01", "us"), np.datetime64("2002-02-02", "us")]
        ...     ),
        ...     start_data_datetime_per_subject=DatetimeGenerator(
        ...         [np.datetime64("2021-01-01", "us"), np.datetime64("2022-02-02", "us")]
        ...     ),
        ...     time_between_birth_and_data_per_subject=PositiveTimeDeltaGenerator(
        ...         [np.timedelta64(20, "Y"), np.timedelta64(30, "Y")]
        ...     ),
        ...     time_between_data_and_death_per_subject=PositiveTimeDeltaGenerator(
        ...         [np.timedelta64(2, "D"), np.timedelta64(10, "D")]
        ...     ),
        ...     time_between_data_events_per_subject=PositiveTimeDeltaGenerator(
        ...         [np.timedelta64(3, "h"), np.timedelta64(20, "m")]
        ...     ),
        ...     num_events_per_subject=PositiveIntGenerator([1, 2, 3]),
        ...     num_measurements_per_event=PositiveIntGenerator([1, 2, 3]),
        ...     num_static_measurements_per_subject=PositiveIntGenerator([2]),
        ...     frac_dynamic_code_occurrences_with_value=ProportionGenerator([0, 0.1, 0.9]),
        ...     frac_static_code_occurrences_with_value=ProportionGenerator([0]),
        ...     static_vocab_size=4,
        ...     dynamic_vocab_size=16,
        ...     frac_subjects_with_death=0.5,
        ... )
        >>> MEDSDataDFGenerator(**kwargs).sample(3, rng)
        shape: (31, 4)
        ┌────────────┬─────────────┬────────────────────────────┬───────────────┐
        │ subject_id ┆ code        ┆ time                       ┆ numeric_value │
        │ ---        ┆ ---         ┆ ---                        ┆ ---           │
        │ i64        ┆ str         ┆ datetime[μs]               ┆ f64           │
        ╞════════════╪═════════════╪════════════════════════════╪═══════════════╡
        │ 0          ┆ static//0   ┆ null                       ┆ null          │
        │ 0          ┆ static//2   ┆ null                       ┆ null          │
        │ 0          ┆ MEDS_BIRTH  ┆ 2002-02-02 00:00:00        ┆ null          │
        │ 0          ┆ dynamic//9  ┆ 2032-02-02 06:36:00        ┆ null          │
        │ 0          ┆ dynamic//9  ┆ 2032-02-02 06:36:00        ┆ null          │
        │ …          ┆ …           ┆ …                          ┆ …             │
        │ 2          ┆ dynamic//12 ┆ 2032-02-02 06:36:00        ┆ null          │
        │ 2          ┆ dynamic//3  ┆ 2032-02-02 07:15:57.171901 ┆ -1.524686     │
        │ 2          ┆ dynamic//9  ┆ 2032-02-02 08:09:01.280550 ┆ null          │
        │ 2          ┆ dynamic//9  ┆ 2032-02-02 08:09:01.280550 ┆ null          │
        │ 2          ┆ dynamic//5  ┆ 2032-02-02 08:09:01.280550 ┆ null          │
        └────────────┴─────────────┴────────────────────────────┴───────────────┘
        >>> DG = MEDSDataDFGenerator(**{**kwargs, "birth_codes_vocab_size": 10, "death_codes_vocab_size": 10})
        >>> DG.sample(3, rng)
        shape: (23, 4)
        ┌────────────┬───────────────┬────────────────────────────┬───────────────┐
        │ subject_id ┆ code          ┆ time                       ┆ numeric_value │
        │ ---        ┆ ---           ┆ ---                        ┆ ---           │
        │ i64        ┆ str           ┆ datetime[μs]               ┆ f64           │
        ╞════════════╪═══════════════╪════════════════════════════╪═══════════════╡
        │ 0          ┆ static//2     ┆ null                       ┆ null          │
        │ 0          ┆ static//3     ┆ null                       ┆ null          │
        │ 0          ┆ MEDS_BIRTH//4 ┆ 2002-02-02 00:00:00        ┆ null          │
        │ 0          ┆ dynamic//2    ┆ 2022-02-01 20:24:00        ┆ -0.673302     │
        │ 0          ┆ dynamic//11   ┆ 2022-02-01 20:24:00        ┆ null          │
        │ …          ┆ …             ┆ …                          ┆ …             │
        │ 2          ┆ dynamic//2    ┆ 2032-02-02 06:36:00        ┆ -0.745856     │
        │ 2          ┆ dynamic//4    ┆ 2032-02-02 07:00:45.528973 ┆ -0.400115     │
        │ 2          ┆ dynamic//15   ┆ 2032-02-02 07:02:59.397283 ┆ null          │
        │ 2          ┆ dynamic//10   ┆ 2032-02-02 07:02:59.397283 ┆ null          │
        │ 2          ┆ MEDS_DEATH//3 ┆ 2032-02-12 07:02:59.397283 ┆ null          │
        └────────────┴───────────────┴────────────────────────────┴───────────────┘

        Errors are thrown in various validation settings:

        >>> MEDSDataDFGenerator(**{**kwargs, "dynamic_vocab_size": 0})
        Traceback (most recent call last):
            ...
        ValueError: dynamic_vocab_size must be a positive integer.
        >>> MEDSDataDFGenerator(**{**kwargs, "static_vocab_size": 0})
        Traceback (most recent call last):
            ...
        ValueError: static_vocab_size must be a positive integer.
        >>> MEDSDataDFGenerator(**{**kwargs, "frac_subjects_with_death": 1.1})
        Traceback (most recent call last):
            ...
        ValueError: frac_subjects_with_death must be a proportion.
        >>> MEDSDataDFGenerator(**{**kwargs, "frac_subjects_with_birth": -0.1})
        Traceback (most recent call last):
            ...
        ValueError: frac_subjects_with_birth must be a proportion.
        >>> MEDSDataDFGenerator(**{**kwargs, "birth_codes_vocab_size": -1})
        Traceback (most recent call last):
            ...
        ValueError: birth_codes_vocab_size must be a non-negative integer.
        >>> MEDSDataDFGenerator(**{**kwargs, "death_codes_vocab_size": -1})
        Traceback (most recent call last):
            ...
        ValueError: death_codes_vocab_size must be a non-negative integer.
        >>> MEDSDataDFGenerator(**{
        ...     **kwargs, "birth_datetime_per_subject": None, "frac_subjects_with_birth": 1
        ... })
        Traceback (most recent call last):
            ...
        ValueError: If birth_datetime_per_subject is None, frac_subjects_with_birth must be 0.
        >>> MEDSDataDFGenerator(**{**kwargs, "birth_codes_vocab_size": 0})
        Traceback (most recent call last):
            ...
        ValueError: If there are births, there must be at least one birth code
        >>> MEDSDataDFGenerator(**{
        ...     **kwargs, "time_between_data_and_death_per_subject": None, "frac_subjects_with_death": 1
        ... })
        Traceback (most recent call last):
            ...
        ValueError: If time_between_data_and_death_per_subject is None, frac_subjects_with_death must be 0.
        >>> MEDSDataDFGenerator(**{**kwargs, "death_codes_vocab_size": 0})
        Traceback (most recent call last):
            ...
        ValueError: If there are deaths, there must be at least one death code
    """

    birth_datetime_per_subject: DatetimeGenerator | None
    start_data_datetime_per_subject: DatetimeGenerator
    time_between_birth_and_data_per_subject: PositiveTimeDeltaGenerator | None
    time_between_data_and_death_per_subject: PositiveTimeDeltaGenerator | None
    time_between_data_events_per_subject: PositiveTimeDeltaGenerator

    num_events_per_subject: PositiveIntGenerator
    num_measurements_per_event: PositiveIntGenerator
    num_static_measurements_per_subject: PositiveIntGenerator
    frac_dynamic_code_occurrences_with_value: ProportionGenerator
    frac_static_code_occurrences_with_value: ProportionGenerator

    static_vocab_size: POSITIVE_INT
    dynamic_vocab_size: POSITIVE_INT
    frac_subjects_with_death: PROPORTION
    frac_subjects_with_birth: PROPORTION = 1
    birth_codes_vocab_size: NON_NEGATIVE_INT = 1
    death_codes_vocab_size: NON_NEGATIVE_INT = 1

    def __post_init__(self):
        if not is_POSITIVE_INT(self.dynamic_vocab_size):
            raise ValueError("dynamic_vocab_size must be a positive integer.")
        if not is_POSITIVE_INT(self.static_vocab_size):
            raise ValueError("static_vocab_size must be a positive integer.")
        if not is_PROPORTION(self.frac_subjects_with_death):
            raise ValueError("frac_subjects_with_death must be a proportion.")
        if not is_PROPORTION(self.frac_subjects_with_birth):
            raise ValueError("frac_subjects_with_birth must be a proportion.")
        if not is_NON_NEGATIVE_INT(self.birth_codes_vocab_size):
            raise ValueError("birth_codes_vocab_size must be a non-negative integer.")
        if not is_NON_NEGATIVE_INT(self.death_codes_vocab_size):
            raise ValueError("death_codes_vocab_size must be a non-negative integer.")

        if not self.has_births:
            if self.frac_subjects_with_birth != 0:
                raise ValueError("If birth_datetime_per_subject is None, frac_subjects_with_birth must be 0.")
        elif self.birth_codes_vocab_size == 0:
            raise ValueError("If there are births, there must be at least one birth code")

        if not self.has_deaths:
            if self.frac_subjects_with_death != 0:
                raise ValueError(
                    "If time_between_data_and_death_per_subject is None, frac_subjects_with_death must be 0."
                )
        elif self.death_codes_vocab_size == 0:
            raise ValueError("If there are deaths, there must be at least one death code")

    @property
    def birth_codes(self):
        if self.birth_codes_vocab_size == 1:
            return [birth_code]
        return [f"{birth_code}//{i}" for i in range(self.birth_codes_vocab_size)]

    @property
    def death_codes(self):
        if self.death_codes_vocab_size == 1:
            return [death_code]
        return [f"{death_code}//{i}" for i in range(self.death_codes_vocab_size)]

    @property
    def has_births(self) -> bool:
        return self.birth_datetime_per_subject is not None

    @property
    def has_deaths(self) -> bool:
        return self.time_between_data_and_death_per_subject is not None

    @property
    def _subject_specific_gens(self) -> list[DiscreteGenerator]:
        out = [
            ("num_static_measurements", self.num_static_measurements_per_subject),
            ("num_events", self.num_events_per_subject),
            ("start_data_datetime", self.start_data_datetime_per_subject),
            ("time_between_data_events", self.time_between_data_events_per_subject),
        ]
        if self.has_births:
            out.append(("birth_datetime", self.birth_datetime_per_subject))
            out.append(
                (
                    "time_between_birth_and_data",
                    self.time_between_birth_and_data_per_subject,
                )
            )
        if self.has_deaths:
            out.append(
                (
                    "time_between_data_and_death",
                    self.time_between_data_and_death_per_subject,
                )
            )
        return out

    def _sample_code_val(
        self,
        size: int,
        vocab_size: int,
        value_props: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple:
        codes = rng.choice(vocab_size, size=size)
        value_obs_p = value_props[codes]
        value_obs = rng.random(size=size) < value_obs_p
        value_num = rng.normal(size=size)
        values = np.where(value_obs, value_num, None)
        return codes, values

    def sample(self, N_subjects: int, rng: np.random.Generator) -> pl.DataFrame:
        dynamic_codes_value_props = self.frac_dynamic_code_occurrences_with_value.rvs(
            self.dynamic_vocab_size, rng
        )
        static_codes_value_props = self.frac_static_code_occurrences_with_value.rvs(
            self.static_vocab_size, rng
        )

        per_subject_samples = {}
        for n, gen in self._subject_specific_gens:
            try:
                per_subject_samples[n] = gen.rvs(N_subjects, rng)
            except Exception as e:  # pragma: no cover
                raise ValueError(f"Failed to generate {n}") from e

        num_events_per_subject = per_subject_samples["num_events"]
        per_subject_samples["num_measurements_per_event"] = np.split(
            self.num_measurements_per_event.rvs(sum(num_events_per_subject), rng),
            np.cumsum(num_events_per_subject),
        )[:-1]

        num_static_measurements = per_subject_samples["num_static_measurements"]
        static_codes, static_values = self._sample_code_val(
            size=sum(num_static_measurements),
            vocab_size=self.static_vocab_size,
            value_props=static_codes_value_props,
            rng=rng,
        )

        per_subject_samples["static_codes"] = np.split(static_codes, np.cumsum(num_static_measurements))[:-1]
        per_subject_samples["static_values"] = np.split(static_values, np.cumsum(num_static_measurements))[
            :-1
        ]

        if self.has_births:
            birth_code_obs_per_subject = rng.binomial(n=1, p=self.frac_subjects_with_birth, size=N_subjects)
            birth_code_per_subject = rng.choice(self.birth_codes, size=N_subjects)
            per_subject_samples["birth_code"] = np.where(
                birth_code_obs_per_subject, birth_code_per_subject, None
            )

        if self.has_deaths:
            death_code_obs_per_subject = rng.binomial(n=1, p=self.frac_subjects_with_death, size=N_subjects)
            death_code_per_subject = rng.choice(self.death_codes, size=N_subjects)
            per_subject_samples["death_code"] = np.where(
                death_code_obs_per_subject, death_code_per_subject, None
            )

        dataset = {}
        dataset[subject_id_field] = []
        dataset[code_field] = []
        dataset[time_field] = []
        dataset[numeric_value_field] = []

        for subject in range(N_subjects):
            subject_samples = {k: v[subject] for k, v in per_subject_samples.items()}

            num_static_measurements = subject_samples["num_static_measurements"]
            static_codes = subject_samples["static_codes"]
            static_values = subject_samples["static_values"]

            dataset[subject_id_field].extend([subject] * num_static_measurements)
            dataset[time_field].extend([None] * num_static_measurements)
            dataset[code_field].extend(f"static//{i}" for i in static_codes)
            dataset[numeric_value_field].extend(static_values)

            if subject_samples.get("birth_code", False):
                birth_datetime = subject_samples["birth_datetime"].astype("datetime64[us]")
                dataset[subject_id_field].append(subject)
                dataset[time_field].append(birth_datetime)
                dataset[code_field].append(subject_samples["birth_code"])
                dataset[numeric_value_field].append(None)

                event_datetime = birth_datetime + subject_samples["time_between_birth_and_data"].astype(
                    "timedelta64[us]"
                )
            else:
                event_datetime = subject_samples["start_data_datetime"]

            num_events = subject_samples["num_events"]
            sec_between_events = rng.exponential(
                subject_samples["time_between_data_events"] / np.timedelta64(1, "s"),
                size=num_events,
            )
            timedeltas = [np.timedelta64(int(s * 1e6), "us") for s in sec_between_events]

            for n, timedelta in zip(subject_samples["num_measurements_per_event"], timedeltas, strict=False):
                codes, values = self._sample_code_val(
                    size=n,
                    vocab_size=self.dynamic_vocab_size,
                    value_props=dynamic_codes_value_props,
                    rng=rng,
                )

                dataset[code_field].extend([f"dynamic//{i}" for i in codes])
                dataset[subject_id_field].extend([subject] * n)
                dataset[time_field].extend([event_datetime] * n)
                dataset[numeric_value_field].extend(values)

                event_datetime += timedelta

            last_event_datetime = dataset[time_field][-1]
            if subject_samples.get("death_code", False):
                time_between_data_and_death = subject_samples["time_between_data_and_death"]
                dataset[subject_id_field].append(subject)
                dataset[time_field].append(last_event_datetime + time_between_data_and_death)
                dataset[code_field].append(subject_samples["death_code"])
                dataset[numeric_value_field].append(None)

        dataset[time_field] = np.array(dataset[time_field], dtype="datetime64[us]")

        return pl.DataFrame(dataset)


@dataclass
class MEDSDatasetGenerator:
    """A class to generate whole MEDS datasets, including core data and metadata.

    Note that these datasets are _not_ meaningful datasets, but rather are just random data for use in testing
    or benchmarking purposes.

    Args:
        data_generator: The data generator to use.
        shard_size: The number of subjects per shard. If None, the dataset will be split into two shards.
        train_frac: The fraction of subjects to use for training.
        tuning_frac: The fraction of subjects to use for tuning. If None, the remaining fraction will be used.
            If both tuning_frac and held_out_frac are None, the remaining fraction will be split evenly
            between the two.
        held_out_frac: The fraction of subjects to use for the held-out set. If None, the remaining fraction
            will be used. If both tuning_frac and held_out_frac are None, the remaining fraction will be split
            evenly between the two.
        dataset_name: The name of the dataset. If None, a default name will be generated.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> data_df_gen = MEDSDataDFGenerator(
        ...     birth_datetime_per_subject=DatetimeGenerator(
        ...         [np.datetime64("2001-01-01", "us"), np.datetime64("2002-02-02", "us")]
        ...     ),
        ...     start_data_datetime_per_subject=DatetimeGenerator(
        ...         [np.datetime64("2021-01-01", "us"), np.datetime64("2022-02-02", "us")]
        ...     ),
        ...     time_between_birth_and_data_per_subject=PositiveTimeDeltaGenerator(
        ...         [np.timedelta64(20, "Y"), np.timedelta64(30, "Y")]
        ...     ),
        ...     time_between_data_and_death_per_subject=PositiveTimeDeltaGenerator(
        ...         [np.timedelta64(2, "D"), np.timedelta64(10, "D")]
        ...     ),
        ...     time_between_data_events_per_subject=PositiveTimeDeltaGenerator(
        ...         [np.timedelta64(3, "h"), np.timedelta64(20, "m")]
        ...     ),
        ...     num_events_per_subject=PositiveIntGenerator([1, 2, 3]),
        ...     num_measurements_per_event=PositiveIntGenerator([1, 2, 3]),
        ...     num_static_measurements_per_subject=PositiveIntGenerator([2]),
        ...     frac_dynamic_code_occurrences_with_value=ProportionGenerator([0, 0.1, 0.9]),
        ...     frac_static_code_occurrences_with_value=ProportionGenerator([0]),
        ...     static_vocab_size=4,
        ...     dynamic_vocab_size=16,
        ...     frac_subjects_with_death=0.5,
        ... )
        >>> G = MEDSDatasetGenerator(data_generator=data_df_gen, shard_size=3, dataset_name="MEDS_Sample")
        >>> dataset = G.sample(10, rng)
        >>> for k, v in dataset.dataset_metadata.items():
        ...     if k == "etl_version":
        ...         print(f"{k}: {v.replace(__version__, '...')}") # This is dynamic so we omit it here.
        ...     elif k == "created_at":
        ...         print(f"{k}: ...") # This is dynamic so we omit it here.
        ...     else:
        ...         print(f"{k}: {v}")
        dataset_name: MEDS_Sample
        dataset_version: 0.0.1
        etl_name: meds_testing_helpers
        etl_version: ...
        meds_version: 0.3.3
        created_at: ...
        extension_columns: []
        >>> dataset._pl_code_metadata # This is always empty for now as these codes are meaningless.
        shape: (0, 3)
        ┌──────┬─────────────┬──────────────┐
        │ code ┆ description ┆ parent_codes │
        │ ---  ┆ ---         ┆ ---          │
        │ str  ┆ str         ┆ list[str]    │
        ╞══════╪═════════════╪══════════════╡
        └──────┴─────────────┴──────────────┘
        >>> dataset._pl_subject_splits
        shape: (10, 2)
        ┌────────────┬──────────┐
        │ subject_id ┆ split    │
        │ ---        ┆ ---      │
        │ i64        ┆ str      │
        ╞════════════╪══════════╡
        │ 4          ┆ train    │
        │ 0          ┆ train    │
        │ 1          ┆ train    │
        │ 9          ┆ train    │
        │ 7          ┆ train    │
        │ 2          ┆ train    │
        │ 6          ┆ train    │
        │ 8          ┆ train    │
        │ 5          ┆ tuning   │
        │ 3          ┆ held_out │
        └────────────┴──────────┘
        >>> len(dataset.data_shards)
        3
        >>> dataset._pl_shards["0"]
        shape: (31, 4)
        ┌────────────┬─────────────┬────────────────────────────┬───────────────┐
        │ subject_id ┆ code        ┆ time                       ┆ numeric_value │
        │ ---        ┆ ---         ┆ ---                        ┆ ---           │
        │ i64        ┆ str         ┆ datetime[μs]               ┆ f64           │
        ╞════════════╪═════════════╪════════════════════════════╪═══════════════╡
        │ 0          ┆ static//0   ┆ null                       ┆ null          │
        │ 0          ┆ static//2   ┆ null                       ┆ null          │
        │ 0          ┆ MEDS_BIRTH  ┆ 2002-02-02 00:00:00        ┆ null          │
        │ 0          ┆ dynamic//9  ┆ 2032-02-02 06:36:00        ┆ null          │
        │ 0          ┆ dynamic//9  ┆ 2032-02-02 06:36:00        ┆ null          │
        │ …          ┆ …           ┆ …                          ┆ …             │
        │ 2          ┆ dynamic//12 ┆ 2032-02-02 06:36:00        ┆ null          │
        │ 2          ┆ dynamic//3  ┆ 2032-02-02 07:15:57.171901 ┆ -1.524686     │
        │ 2          ┆ dynamic//9  ┆ 2032-02-02 08:09:01.280550 ┆ null          │
        │ 2          ┆ dynamic//9  ┆ 2032-02-02 08:09:01.280550 ┆ null          │
        │ 2          ┆ dynamic//5  ┆ 2032-02-02 08:09:01.280550 ┆ null          │
        └────────────┴─────────────┴────────────────────────────┴───────────────┘
        >>> dataset._pl_shards["1"]
        shape: (22, 4)
        ┌────────────┬─────────────┬────────────────────────────┬───────────────┐
        │ subject_id ┆ code        ┆ time                       ┆ numeric_value │
        │ ---        ┆ ---         ┆ ---                        ┆ ---           │
        │ i64        ┆ str         ┆ datetime[μs]               ┆ f64           │
        ╞════════════╪═════════════╪════════════════════════════╪═══════════════╡
        │ 3          ┆ static//2   ┆ null                       ┆ null          │
        │ 3          ┆ static//3   ┆ null                       ┆ null          │
        │ 3          ┆ MEDS_BIRTH  ┆ 2002-02-02 00:00:00        ┆ null          │
        │ 3          ┆ dynamic//10 ┆ 2022-02-01 20:24:00        ┆ null          │
        │ 3          ┆ dynamic//12 ┆ 2022-02-01 20:24:00        ┆ null          │
        │ …          ┆ …           ┆ …                          ┆ …             │
        │ 5          ┆ dynamic//15 ┆ 2032-02-02 06:36:00        ┆ -0.526515     │
        │ 5          ┆ dynamic//4  ┆ 2032-02-02 06:36:00        ┆ -1.264493     │
        │ 5          ┆ dynamic//10 ┆ 2032-02-02 06:37:04.199317 ┆ null          │
        │ 5          ┆ dynamic//8  ┆ 2032-02-02 06:41:31.716960 ┆ -2.019266     │
        │ 5          ┆ dynamic//4  ┆ 2032-02-02 06:41:31.716960 ┆ 0.420513      │
        └────────────┴─────────────┴────────────────────────────┴───────────────┘
        >>> dataset._pl_shards["2"]
        shape: (24, 4)
        ┌────────────┬────────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ code       ┆ time                ┆ numeric_value │
        │ ---        ┆ ---        ┆ ---                 ┆ ---           │
        │ i64        ┆ str        ┆ datetime[μs]        ┆ f64           │
        ╞════════════╪════════════╪═════════════════════╪═══════════════╡
        │ 6          ┆ static//3  ┆ null                ┆ null          │
        │ 6          ┆ static//0  ┆ null                ┆ null          │
        │ 6          ┆ MEDS_BIRTH ┆ 2001-01-01 00:00:00 ┆ null          │
        │ 6          ┆ dynamic//5 ┆ 2020-12-31 20:24:00 ┆ null          │
        │ 6          ┆ dynamic//1 ┆ 2020-12-31 20:24:00 ┆ null          │
        │ …          ┆ …          ┆ …                   ┆ …             │
        │ 8          ┆ dynamic//3 ┆ 2022-02-01 20:24:00 ┆ null          │
        │ 9          ┆ static//3  ┆ null                ┆ null          │
        │ 9          ┆ static//0  ┆ null                ┆ null          │
        │ 9          ┆ MEDS_BIRTH ┆ 2001-01-01 00:00:00 ┆ null          │
        │ 9          ┆ dynamic//7 ┆ 2031-01-01 06:36:00 ┆ null          │
        └────────────┴────────────┴─────────────────────┴───────────────┘

        You can omit subject splits:

        >>> G = MEDSDatasetGenerator(data_generator=data_df_gen, shard_size=3, train_frac=None)
        >>> dataset = G.sample(10, rng)
        >>> dataset.subject_splits is None
        True

        Errors are thrown in various validation settings.

        >>> MEDSDatasetGenerator(data_generator=data_df_gen, shard_size=0)
        Traceback (most recent call last):
            ...
        ValueError: shard_size must be a positive integer; got 0
        >>> MEDSDatasetGenerator(data_generator=data_df_gen, train_frac=None, tuning_frac=0.5)
        Traceback (most recent call last):
            ...
        ValueError: If train_frac is None, tuning_frac and held_out_frac must be None.
        >>> MEDSDatasetGenerator(data_generator=data_df_gen, train_frac=1.1)
        Traceback (most recent call last):
            ...
        ValueError: train_frac must be between 0 and 1; got 1.1
        >>> MEDSDatasetGenerator(data_generator=data_df_gen, train_frac=0.5, tuning_frac=-0.1)
        Traceback (most recent call last):
            ...
        ValueError: tuning_frac must be between 0 and 1; got -0.1
        >>> MEDSDatasetGenerator(data_generator=data_df_gen, train_frac=0.5, held_out_frac=-0.1)
        Traceback (most recent call last):
            ...
        ValueError: held_out_frac must be between 0 and 1; got -0.1
        >>> MEDSDatasetGenerator(
        ...     data_generator=data_df_gen, train_frac=0.5, held_out_frac=0.5, tuning_frac=0.5
        ... )
        Traceback (most recent call last):
            ...
        ValueError: The sum of train_frac, tuning_frac, and held_out_frac must be 1. Got 0.5 + 0.5 + 0.5 = 1.5
    """

    data_generator: MEDSDataDFGenerator
    shard_size: POSITIVE_INT | None = None
    train_frac: PROPORTION | None = 0.8
    tuning_frac: PROPORTION | None = None
    held_out_frac: PROPORTION | None = None
    dataset_name: str | None = None

    @property
    def has_splits(self) -> bool:
        return self.train_frac is not None

    def __post_init__(self):
        if self.shard_size is not None and not is_POSITIVE_INT(self.shard_size):
            raise ValueError(f"shard_size must be a positive integer; got {self.shard_size}")

        if not self.has_splits:
            if self.tuning_frac is not None or self.held_out_frac is not None:
                raise ValueError("If train_frac is None, tuning_frac and held_out_frac must be None.")
        else:
            if not is_PROPORTION(self.train_frac):
                raise ValueError(f"train_frac must be between 0 and 1; got {self.train_frac}")

            if self.tuning_frac is None and self.held_out_frac is None:
                leftover = 1 - self.train_frac
                self.tuning_frac = round(leftover / 2, 4)
                self.held_out_frac = round(leftover / 2, 4)
            elif self.tuning_frac is None:
                self.tuning_frac = 1 - self.train_frac - self.held_out_frac
            elif self.held_out_frac is None:
                self.held_out_frac = 1 - self.train_frac - self.tuning_frac

            if not is_PROPORTION(self.tuning_frac):
                raise ValueError(f"tuning_frac must be between 0 and 1; got {self.tuning_frac}")
            if not is_PROPORTION(self.held_out_frac):
                raise ValueError(f"held_out_frac must be between 0 and 1; got {self.held_out_frac}")

            if self.train_frac + self.tuning_frac + self.held_out_frac != 1:
                raise ValueError(
                    "The sum of train_frac, tuning_frac, and held_out_frac must be 1. Got "
                    f"{self.train_frac} + {self.tuning_frac} + {self.held_out_frac} = "
                    f"{self.train_frac + self.tuning_frac + self.held_out_frac}"
                )

        if self.dataset_name is None:
            self.dataset_name = f"MEDS_Sample_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    def sample(self, N_subjects: int, rng: np.random.Generator) -> MEDSDataset:
        n_shards = N_subjects // self.shard_size if self.shard_size is not None else 2
        subjects_per_shard = N_subjects // n_shards
        shard_sizes = [subjects_per_shard] * (n_shards - 1) + [
            N_subjects - subjects_per_shard * (n_shards - 1)
        ]

        data_shards = {}
        total_subjects = 0
        for i, size in enumerate(shard_sizes):
            data_shards[str(i)] = self.data_generator.sample(size, rng).with_columns(
                (pl.col(subject_id_field) + total_subjects).alias(subject_id_field)
            )
            total_subjects += size

        dataset_metadata = DatasetMetadata(
            dataset_name=self.dataset_name,
            dataset_version="0.0.1",
            etl_name=__package_name__,
            etl_version=__version__,
            meds_version=meds_version,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            extension_columns=[],
        )

        code_metadata = pl.DataFrame(
            {
                code_field: pl.Series([], dtype=pl.Utf8),
                description_field: pl.Series([], dtype=pl.Utf8),
                parent_codes_field: pl.Series([], dtype=pl.List(pl.Utf8)),
            }
        )

        if self.has_splits:
            subjects = list(range(N_subjects))
            rng.shuffle(subjects)
            N_train = int(N_subjects * self.train_frac)
            N_tuning = int(N_subjects * self.tuning_frac)
            N_held_out = N_subjects - N_train - N_tuning

            split = [train_split] * N_train + [tuning_split] * N_tuning + [held_out_split] * N_held_out
            subject_splits = pl.DataFrame(
                {
                    subject_id_field: pl.Series(subjects, dtype=pl.Int64),
                    "split": pl.Series(split, dtype=pl.Utf8),
                }
            )
        else:
            subject_splits = None

        return MEDSDataset(
            data_shards=data_shards,
            dataset_metadata=dataset_metadata,
            code_metadata=code_metadata,
            subject_splits=subject_splits,
        )


@hydra.main(version_base=None, config_path=str(GEN_YAML.parent), config_name=GEN_YAML.stem)
def main(cfg: DictConfig):
    """Generate a dataset of the specified size."""

    output_dir = Path(cfg.output_dir)

    if output_dir.exists():
        if output_dir.is_file():
            raise ValueError("Output directory is a file; expected a directory.")
        if cfg.do_overwrite:
            logger.warning("Output directory already exists. Overwriting.")
            shutil.rmtree(output_dir)
        elif (output_dir / "data").exists() or (output_dir / "metadata").exists():
            contents = [f"  - {p.relative_to(output_dir)}" for p in output_dir.rglob("*")]
            contents_str = "\n".join(contents)
            raise ValueError(
                f"Output directory is not empty! use --do-overwrite to overwrite. Contents:\n{contents_str}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    G = hydra.utils.instantiate(cfg.dataset_spec)
    rng = np.random.default_rng(cfg.seed)

    logger.info(f"Generating dataset with {cfg.N_subjects} subjects.")
    dataset = G.sample(cfg.N_subjects, rng)

    logger.info(f"Saving dataset to root directory {output_dir.resolve()!s}.")
    dataset.write(output_dir)

    logger.info("Done.")
