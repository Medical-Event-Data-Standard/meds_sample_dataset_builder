import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from meds import (
    DatasetMetadata,
    code_field,
    code_metadata_filepath,
    code_metadata_schema,
    data_schema,
    data_subdirectory,
    dataset_metadata_filepath,
    description_field,
    label_schema,
    numeric_value_field,
    parent_codes_field,
    prediction_time_field,
    subject_id_field,
    subject_split_schema,
    subject_splits_filepath,
    time_field,
)
from yaml import load as load_yaml

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader

logger = logging.getLogger(__name__)


SHARDED_DF_T = dict[str, pl.DataFrame]


class MEDSDataset:
    """A minimal helper class for working with MEDS Datasets intended for use in testing, not production.

    This class is intended to be used for testing and development purposes only. It is not intended to be used
    in production code, and as such has only been optimized for small datasets, and may not support all
    aspects of the MEDS schema.

    Attributes:
        root_dir: The root directory of the dataset, if provided. If specified, data will be read from this
            root directory upon access, not stored in memory. If not provided, the below parameters must be
            provided upon initialization.
        data_shards: A dictionary of data shards, where the keys are the shard names and the values are the
            data tables. Upon access of this attribute, the data will be returned as pyarrow tables. Upon
            specification in the constructor, polars dataframes are expected instead.
        dataset_metadata: The metadata for the dataset, stored as a DatasetMetadata object (which is just a
            type-annotated dictionary).
        code_metadata: The metadata for the codes. Upon access of this attribute, the data will be returned as
            a pyarrow table. Upon specification in the constructor, a polars dataframe is expected instead.
        subject_splits: The subject splits for the dataset. Optional. Upon access of this attribute, the data
            will be returned as a pyarrow table. Upon specification in the constructor, a polars dataframe is
            expected instead. If not specified for an otherwise valid dataset, `None` will be returned.
        task_labels: Optionally, you can also include task labels. These are not formally connected to a
            particular MEDS dataset (in terms of storage on disk; see
            https://github.com/Medical-Event-Data-Standard/meds/issues/75 for more information), you can also
            track a collection of sharded task labels by task name in this class. If specified, this should be
            a dictionary mapping task name to sharded task files, in the proper MEDS label dataframe format.

    Examples:
        >>> data_shards = {
        ...     "0": pl.DataFrame({"subject_id": [0], "time": [0], "numeric_value": [None], "code": ["A"]}),
        ...     "1": pl.DataFrame({"subject_id": [1], "time": [0], "numeric_value": [1.0], "code": ["B"]}),
        ... }
        >>> dataset_metadata = DatasetMetadata(
        ...     dataset_name="test",
        ...     dataset_version="0.0.1",
        ...     etl_name="foo",
        ...     etl_version="0.0.1",
        ...     meds_version="0.3.3",
        ...     created_at="1/1/2025",
        ...     extension_columns=[],
        ... )
        >>> code_metadata = pl.DataFrame({
        ...     "code": ["A", "B"], "description": ["foo", "bar"],
        ...     "parent_codes": pl.Series([None, None], dtype=pl.List(pl.Utf8)),
        ... })
        >>> subject_splits = None
        >>> task_labels = None
        >>> D = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     subject_splits=subject_splits,
        ...     task_labels=task_labels,
        ... )
        >>> D # doctest: +NORMALIZE_WHITESPACE
        MEDSDataset(data_shards={'0': {'subject_id': [0],
                                       'time': [0],
                                       'numeric_value': [None],
                                       'code': ['A']},
                                 '1': {'subject_id': [1],
                                       'time': [0],
                                       'numeric_value': [1.0],
                                       'code': ['B']}},
                    dataset_metadata={'dataset_name': 'test',
                                      'dataset_version': '0.0.1',
                                      'etl_name': 'foo',
                                      'etl_version': '0.0.1',
                                      'meds_version': '0.3.3',
                                      'created_at': '1/1/2025',
                                      'extension_columns': []},
                    code_metadata={'code': ['A', 'B'],
                                   'description': ['foo', 'bar'],
                                   'parent_codes': [None, None]})
        >>> print(D)
        MEDSDataset:
        dataset_metadata:
          - dataset_name: test
          - dataset_version: 0.0.1
          - etl_name: foo
          - etl_version: 0.0.1
          - meds_version: 0.3.3
          - created_at: 1/1/2025
          - extension_columns: []
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[0]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
          - 1:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["B"]]
            numeric_value: [[1]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: [["A","B"]]
          description: [["foo","bar"]]
          parent_codes: [[null,null]]
        subject_splits: None
        >>> print(D.shard_fps)
        None

        Note that code metadata can be inferred to be empty if not provided:

        >>> print(MEDSDataset(data_shards=data_shards, dataset_metadata=dataset_metadata))
        MEDSDataset:
        dataset_metadata:
          - dataset_name: test
          - dataset_version: 0.0.1
          - etl_name: foo
          - etl_version: 0.0.1
          - meds_version: 0.3.3
          - created_at: 1/1/2025
          - extension_columns: []
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[0]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
          - 1:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["B"]]
            numeric_value: [[1]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: []
          description: []
          parent_codes: []
        subject_splits: None

        Note that when no task labels are provided, their properties are `None`:

        >>> print(D.task_names)
        None
        >>> print(D.task_labels)
        None

        There are also a collection of filepath related properties that are `None` when the root directory is
        not set:

        >>> print(D.task_root_dir)
        None
        >>> print(D.task_names_fp)
        None
        >>> print(D.task_label_fps)
        None
        >>> print(D.dataset_metadata_fp)
        None
        >>> print(D.code_metadata_fp)
        None
        >>> print(D.subject_splits_fp)
        None
        >>> print(D.shard_fps)
        None

        You can save and load datasets from disk in the proper format. Note that equality persists after this
        operation. Note that, when existent, filepath variables are then set as well. But, those parameters
        that depend on files existing when they don't may still be None

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     D2 = D.write(Path(tmpdir))
        ...     assert D == D2
        ...     print(f"repr: {repr(D2).replace(tmpdir, '...')}")
        ...     print(f"str: {str(D2).replace(tmpdir, '...')}")
        ...     print("|")
        ...     print("Filepaths:")
        ...     print(f"  task_root_dir: {str(D2.task_root_dir).replace(tmpdir, '...')}")
        ...     print(f"  task_names_fp: {str(D2.task_names_fp).replace(tmpdir, '...')}")
        ...     print(f"  dataset_metadata_fp: {str(D2.dataset_metadata_fp).replace(tmpdir, '...')}")
        ...     print(f"  code_metadata_fp: {str(D2.code_metadata_fp).replace(tmpdir, '...')}")
        ...     print(f"  subject_splits_fp: {str(D2.subject_splits_fp).replace(tmpdir, '...')}")
        ...     print(f"  shard_fps: {str(D2.shard_fps).replace(tmpdir, '...')}")
        ...     print(f"  task_label_fps: {D2.task_label_fps}")
        repr: MEDSDataset(root_dir=PosixPath('...'))
        str: MEDSDataset:
        stored in root_dir: ...
        dataset_metadata:
          - dataset_name: test
          - dataset_version: 0.0.1
          - etl_name: foo
          - etl_version: 0.0.1
          - meds_version: 0.3.3
          - created_at: 1/1/2025
          - extension_columns: []
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[0]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
          - 1:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["B"]]
            numeric_value: [[1]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: [["A","B"]]
          description: [["foo","bar"]]
          parent_codes: [[null,null]]
        subject_splits: None
        |
        Filepaths:
          task_root_dir: .../task_labels
          task_names_fp: .../task_labels/.task_names.json
          dataset_metadata_fp: .../metadata/dataset.json
          code_metadata_fp: .../metadata/codes.parquet
          subject_splits_fp: .../metadata/subject_splits.parquet
          shard_fps: [PosixPath('.../data/0.parquet'), PosixPath('.../data/1.parquet')]
          task_label_fps: None

        You can also add subject splits to the dataset:

        >>> subject_splits = pl.DataFrame({"subject_id": [0, 1], "split": ["train", "held_out"]})
        >>> D = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     subject_splits=subject_splits,
        ... )
        >>> print(D)
        MEDSDataset:
        dataset_metadata:
          - dataset_name: test
          - dataset_version: 0.0.1
          - etl_name: foo
          - etl_version: 0.0.1
          - meds_version: 0.3.3
          - created_at: 1/1/2025
          - extension_columns: []
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[0]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
          - 1:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["B"]]
            numeric_value: [[1]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: [["A","B"]]
          description: [["foo","bar"]]
          parent_codes: [[null,null]]
        subject_splits:
          pyarrow.Table
          subject_id: int64
          split: string
          ----
          subject_id: [[0,1]]
          split: [["train","held_out"]]

        Here's an example with task labels. Note that, when provided, you can also query task names
        with a dedicated property (otherwise it is null). Task labels can be empty, or have empty shards. Note
        that root-dir related task properties can still be None even if tasks are set.

        >>> task_df_empty = pl.DataFrame(
        ...     {
        ...         "subject_id": [],
        ...         "prediction_time": [],
        ...         "boolean_value": [],
        ...         "integer_value": [],
        ...         "float_value": [],
        ...         "categorical_value": [],
        ...     },
        ...     schema=MEDSDataset.PL_LABEL_SCHEMA
        ... )
        >>> task_df_nonempty = pl.DataFrame(
        ...     {
        ...         "subject_id": [0, 1],
        ...         "prediction_time": [0, 10],
        ...         "boolean_value": [True, False],
        ...         "integer_value": [None, None],
        ...         "float_value": [None, None],
        ...         "categorical_value": [None, None],
        ...     },
        ...     schema=MEDSDataset.PL_LABEL_SCHEMA
        ... )
        >>> task_labels = { # Format is task_name -> shard -> dataframe
        ...     "A": {},
        ...     "B": {"0": task_df_empty},
        ...     "C": {"0": task_df_nonempty},
        ... }
        >>> D = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     task_labels=task_labels,
        ... )
        >>> D # doctest: +NORMALIZE_WHITESPACE
        MEDSDataset(data_shards={'0': {'subject_id': [0],
                                       'time': [0],
                                       'numeric_value': [None],
                                       'code': ['A']},
                                 '1': {'subject_id': [1],
                                       'time': [0],
                                       'numeric_value': [1.0],
                                       'code': ['B']}},
                    dataset_metadata={'dataset_name': 'test',
                                      'dataset_version': '0.0.1',
                                      'etl_name': 'foo',
                                      'etl_version': '0.0.1',
                                      'meds_version': '0.3.3',
                                      'created_at': '1/1/2025',
                                      'extension_columns': []},
                    code_metadata={'code': ['A', 'B'],
                                   'description': ['foo', 'bar'],
                                   'parent_codes': [None, None]},
                    task_labels={'A': {},
                                 'B': {'0': {'subject_id': [],
                                             'prediction_time': [],
                                             'boolean_value': [],
                                             'integer_value': [],
                                             'float_value': [],
                                             'categorical_value': []}},
                                 'C': {'0': {'subject_id': [0, 1],
                                             'prediction_time': [datetime.datetime(1970, 1, 1, 0, 0),
                                                                 datetime.datetime(1970, 1, 1, 0, 0, 0, 10)],
                                             'boolean_value': [True, False],
                                             'integer_value': [None, None],
                                             'float_value': [None, None],
                                             'categorical_value': [None, None]}}})
        >>> print(D)
        MEDSDataset:
        dataset_metadata:
          - dataset_name: test
          - dataset_version: 0.0.1
          - etl_name: foo
          - etl_version: 0.0.1
          - meds_version: 0.3.3
          - created_at: 1/1/2025
          - extension_columns: []
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[0]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
          - 1:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["B"]]
            numeric_value: [[1]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: [["A","B"]]
          description: [["foo","bar"]]
          parent_codes: [[null,null]]
        subject_splits: None
        task labels:
          * A:
          * B:
            - 0:
              pyarrow.Table
              subject_id: int64
              prediction_time: timestamp[us]
              boolean_value: bool
              integer_value: int64
              float_value: double
              categorical_value: string
              ----
              subject_id: [[]]
              prediction_time: [[]]
              boolean_value: [[]]
              integer_value: [[]]
              float_value: [[]]
              categorical_value: []
          * C:
            - 0:
              pyarrow.Table
              subject_id: int64
              prediction_time: timestamp[us]
              boolean_value: bool
              integer_value: int64
              float_value: double
              categorical_value: string
              ----
              subject_id: [[0,1]]
              prediction_time: [[1970-01-01 00:00:00.000000,1970-01-01 00:00:00.000010]]
              boolean_value: [[true,false]]
              integer_value: [[null,null]]
              float_value: [[null,null]]
              categorical_value: [[null,null]]
        >>> D.task_names
        ['A', 'B', 'C']

        Note that as we don't have a root dir, the file path parameters are still `None`

        >>> print(D.task_root_dir)
        None
        >>> print(D.task_label_fps)
        None

        Reading/Writing with task labels works identically as without:

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     D2 = D.write(Path(tmpdir))
        ...     assert D == D2
        ...     print(f"repr: {repr(D2).replace(tmpdir, '...')}")
        ...     print(f"str: {str(D2).replace(tmpdir, '...')}")
        repr: MEDSDataset(root_dir=PosixPath('...'))
        str: MEDSDataset:
        stored in root_dir: ...
        dataset_metadata:
          - dataset_name: test
          - dataset_version: 0.0.1
          - etl_name: foo
          - etl_version: 0.0.1
          - meds_version: 0.3.3
          - created_at: 1/1/2025
          - extension_columns: []
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[0]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
          - 1:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["B"]]
            numeric_value: [[1]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: [["A","B"]]
          description: [["foo","bar"]]
          parent_codes: [[null,null]]
        subject_splits: None
        task labels:
          * A:
          * B:
            - 0:
              pyarrow.Table
              subject_id: int64
              prediction_time: timestamp[us]
              boolean_value: bool
              integer_value: int64
              float_value: double
              categorical_value: string
              ----
              subject_id: [[]]
              prediction_time: [[]]
              boolean_value: [[]]
              integer_value: [[]]
              float_value: [[]]
              categorical_value: []
          * C:
            - 0:
              pyarrow.Table
              subject_id: int64
              prediction_time: timestamp[us]
              boolean_value: bool
              integer_value: int64
              float_value: double
              categorical_value: string
              ----
              subject_id: [[0,1]]
              prediction_time: [[1970-01-01 00:00:00.000000,1970-01-01 00:00:00.000010]]
              boolean_value: [[true,false]]
              integer_value: [[null,null]]
              float_value: [[null,null]]
              categorical_value: [[null,null]]

        Equality is determined by the equality of the data, metadata, code metadata, subject splits, and task
        labels:

        >>> D1 = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     subject_splits=subject_splits
        ... )
        >>> D1 == "foobar"
        False
        >>> D2 = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     subject_splits=subject_splits
        ... )
        >>> D1 == D2
        True
        >>> D2 = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     subject_splits=None,
        ... )
        >>> D1 == D2
        False
        >>> alt_data_shards = {
        ...     "0": pl.DataFrame({"subject_id": [1], "time": [0], "numeric_value": [None], "code": ["A"]}),
        ...     "1": pl.DataFrame({"subject_id": [1], "time": [0], "numeric_value": [1.0], "code": ["B"]}),
        ... }
        >>> D2 = MEDSDataset(
        ...     data_shards=alt_data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     subject_splits=subject_splits
        ... )
        >>> D1 == D2
        False
        >>> alt_dataset_metadata = DatasetMetadata(
        ...     dataset_name="test_2",
        ...     dataset_version="0.0.1",
        ...     etl_name="foo",
        ...     etl_version="0.0.1",
        ...     meds_version="0.3.3",
        ...     created_at="1/1/2025",
        ...     extension_columns=[],
        ... )
        >>> D2 = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=alt_dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     subject_splits=subject_splits
        ... )
        >>> D1 == D2
        False
        >>> alt_code_metadata = pl.DataFrame({
        ...     "code": ["A", "B"], "description": ["bar", "foo"],
        ...     "parent_codes": pl.Series([None, None], dtype=pl.List(pl.Utf8)),
        ... })
        >>> D2 = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=alt_code_metadata,
        ...     subject_splits=subject_splits
        ... )
        >>> D1 == D2
        False
        >>> D2 = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     subject_splits=subject_splits,
        ...     task_labels=task_labels,
        ... )
        >>> D1 == D2
        False

        Errors are raised in a number of circumstances:

        >>> MEDSDataset()
        Traceback (most recent call last):
            ...
        ValueError: data_shards must be provided if root_dir is None
        >>> MEDSDataset(data_shards=data_shards)
        Traceback (most recent call last):
            ...
        ValueError: dataset_metadata must be provided if root_dir is None
    """

    CSV_TS_FORMAT: ClassVar[str] = "%m/%d/%Y, %H:%M:%S"

    PL_DATA_SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        subject_id_field: pl.Int64,
        time_field: pl.Datetime("us"),
        code_field: pl.String,
        numeric_value_field: pl.Float32,
    }

    PL_CODE_METADATA_SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        code_field: pl.String,
        description_field: pl.String,
        parent_codes_field: pl.List(pl.String),
    }

    PL_SUBJECT_SPLIT_SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        subject_id_field: pl.Int64,
        "split": pl.String,
    }

    PL_LABEL_SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        subject_id_field: pl.Int64,
        prediction_time_field: pl.Datetime("us"),
        "boolean_value": pl.Boolean,
        "integer_value": pl.Int64,
        "float_value": pl.Float64,
        "categorical_value": pl.String,
    }

    TIME_FIELDS: ClassVar[set[str]] = {time_field, prediction_time_field}

    TASK_LABELS_SUBDIR: ClassVar[str] = "task_labels"
    TASK_NAMES_FN: ClassVar[str] = ".task_names.json"

    def __init__(
        self,
        root_dir: Path | None = None,
        data_shards: SHARDED_DF_T | None = None,
        dataset_metadata: DatasetMetadata | None = None,
        code_metadata: pl.DataFrame | None = None,
        subject_splits: pl.DataFrame | None = None,
        task_labels: dict[str, SHARDED_DF_T] | None = None,
    ):
        if root_dir is None:
            if data_shards is None:
                raise ValueError("data_shards must be provided if root_dir is None")
            if dataset_metadata is None:
                raise ValueError("dataset_metadata must be provided if root_dir is None")
            if code_metadata is None:
                logger.warning("Inferring empty code metadata as none was provided.")
                code_metadata = pl.DataFrame(
                    {code_field: [], description_field: [], parent_codes_field: []},
                    schema=self.PL_CODE_METADATA_SCHEMA,
                )

        self.root_dir = root_dir
        self.data_shards = data_shards
        self.dataset_metadata = dataset_metadata
        self.code_metadata = code_metadata
        self.subject_splits = subject_splits
        self.task_labels = task_labels

        # These will throw errors if the data is malformed.
        self.data_shards  # noqa: B018
        self.code_metadata  # noqa: B018
        self.subject_splits  # noqa: B018
        self.dataset_metadata  # noqa: B018

    @classmethod
    def parse_csv(cls, csv: str, **schema_updates) -> pl.DataFrame:
        """Parses a CSV string into a MEDS-related dataframe using the provided schema.

        Args:
            csv: The CSV string to parse.
            schema_updates: The schema to use when parsing the CSV, passed as keyword arguments. Note that
                timestamp columns will be read as strings then converted to the requested type using the
                default CSV timestamp format. Schema defaults are drawn from all the available MEDS schemas,
                _without consideration for whether the passed csv is appropriate for that given schema_.

        Returns:
            A polars DataFrame with the parsed data.

        Raises:
            ValueError: If the CSV cannot be read under the provided schema and/or the schema defaults.

        Examples:
            >>> MEDSDataset.parse_csv(
            ...     'subject_id,time,code,numeric_value\\n0,"1/1/2025, 12:00:00",foo,1.0',
            ...     subject_id=pl.Int64, time=pl.Datetime("us"), code=pl.String, numeric_value=pl.Float32
            ... )
            shape: (1, 4)
            ┌────────────┬─────────────────────┬──────┬───────────────┐
            │ subject_id ┆ time                ┆ code ┆ numeric_value │
            │ ---        ┆ ---                 ┆ ---  ┆ ---           │
            │ i64        ┆ datetime[μs]        ┆ str  ┆ f32           │
            ╞════════════╪═════════════════════╪══════╪═══════════════╡
            │ 0          ┆ 2025-01-01 12:00:00 ┆ foo  ┆ 1.0           │
            └────────────┴─────────────────────┴──────┴───────────────┘

            Note that schema defaults are sourced from all MEDS schemas:

            >>> MEDSDataset.parse_csv(
            ...     'subject_id,time,code,numeric_value\\n0,"1/1/2025, 12:00:00",foo,1.0',
            ... )
            shape: (1, 4)
            ┌────────────┬─────────────────────┬──────┬───────────────┐
            │ subject_id ┆ time                ┆ code ┆ numeric_value │
            │ ---        ┆ ---                 ┆ ---  ┆ ---           │
            │ i64        ┆ datetime[μs]        ┆ str  ┆ f32           │
            ╞════════════╪═════════════════════╪══════╪═══════════════╡
            │ 0          ┆ 2025-01-01 12:00:00 ┆ foo  ┆ 1.0           │
            └────────────┴─────────────────────┴──────┴───────────────┘
            >>> MEDSDataset.parse_csv(
            ...     'code,description,parent_codes\\nfoo,foobar code,"bar,baz"',
            ... )
            shape: (1, 3)
            ┌──────┬─────────────┬──────────────┐
            │ code ┆ description ┆ parent_codes │
            │ ---  ┆ ---         ┆ ---          │
            │ str  ┆ str         ┆ list[str]    │
            ╞══════╪═════════════╪══════════════╡
            │ foo  ┆ foobar code ┆ ["bar,baz"]  │
            └──────┴─────────────┴──────────────┘
            >>> MEDSDataset.parse_csv(
            ...     'subject_id,split\\n0,train\\n1,test',
            ... )
            shape: (2, 2)
            ┌────────────┬───────┐
            │ subject_id ┆ split │
            │ ---        ┆ ---   │
            │ i64        ┆ str   │
            ╞════════════╪═══════╡
            │ 0          ┆ train │
            │ 1          ┆ test  │
            └────────────┴───────┘
            >>> MEDSDataset.parse_csv(
            ...     'subject_id,prediction_time,boolean_value\\n0,"1/1/2025, 12:00:00",False',
            ... )
            shape: (1, 3)
            ┌────────────┬─────────────────────┬───────────────┐
            │ subject_id ┆ prediction_time     ┆ boolean_value │
            │ ---        ┆ ---                 ┆ ---           │
            │ i64        ┆ datetime[μs]        ┆ bool          │
            ╞════════════╪═════════════════════╪═══════════════╡
            │ 0          ┆ 2025-01-01 12:00:00 ┆ false         │
            └────────────┴─────────────────────┴───────────────┘

            Note that columns are not verified to come from a single MEDS schema:

            >>> MEDSDataset.parse_csv(
            ...     'subject_id,prediction_time,split\\n0,"1/1/2025, 12:00:00",train',
            ... )
            shape: (1, 3)
            ┌────────────┬─────────────────────┬───────┐
            │ subject_id ┆ prediction_time     ┆ split │
            │ ---        ┆ ---                 ┆ ---   │
            │ i64        ┆ datetime[μs]        ┆ str   │
            ╞════════════╪═════════════════════╪═══════╡
            │ 0          ┆ 2025-01-01 12:00:00 ┆ train │
            └────────────┴─────────────────────┴───────┘

            Columns from MEDS schemas can also be overwritten with the schema updates keyword arguments:

            >>> MEDSDataset.parse_csv(
            ...     'subject_id,prediction_time,parent_codes\\n0,"1/1/2025, 12:00:00",train',
            ...     subject_id=pl.String, prediction_time=pl.String, parent_codes=pl.String
            ... )
            shape: (1, 3)
            ┌────────────┬────────────────────┬──────────────┐
            │ subject_id ┆ prediction_time    ┆ parent_codes │
            │ ---        ┆ ---                ┆ ---          │
            │ str        ┆ str                ┆ str          │
            ╞════════════╪════════════════════╪══════════════╡
            │ 0          ┆ 1/1/2025, 12:00:00 ┆ train        │
            └────────────┴────────────────────┴──────────────┘

            Columns can also be inferred from the provided CSV, though only for columns not in a dedicated
            MEDS schema:

            >>> MEDSDataset.parse_csv(
            ...     'subject_id,time,code_2,numeric_value\\n0,"1/1/2025, 12:00:00",foo,1.0',
            ... )
            shape: (1, 4)
            ┌────────────┬─────────────────────┬────────┬───────────────┐
            │ subject_id ┆ time                ┆ code_2 ┆ numeric_value │
            │ ---        ┆ ---                 ┆ ---    ┆ ---           │
            │ i64        ┆ datetime[μs]        ┆ str    ┆ f32           │
            ╞════════════╪═════════════════════╪════════╪═══════════════╡
            │ 0          ┆ 2025-01-01 12:00:00 ┆ foo    ┆ 1.0           │
            └────────────┴─────────────────────┴────────┴───────────────┘

            Errors are raised when the schema is incomplete, inaccurate, or the CSV is malformed:

            >>> MEDSDataset.parse_csv(
            ...     'subject_id,time,code,numeric_value\\n0,"1/1/2025, 12:00:00",foo,foo',
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Failed to read:...
            >>> MEDSDataset.parse_csv(123)
            Traceback (most recent call last):
                ...
            ValueError: csv must be a string; got <class 'int'>
        """

        if not isinstance(csv, str):
            raise ValueError(f"csv must be a string; got {type(csv)}")

        read_schema = {}
        time_schema = {}
        has_parent_codes = False

        cols = csv.split("\n")[0].split(",")
        for col in cols:
            do_retype_time = col in cls.TIME_FIELDS
            do_retype_parent_codes = col == parent_codes_field
            if col in schema_updates:
                read_schema[col] = schema_updates[col]
                if col in cls.TIME_FIELDS and not schema_updates[col].is_temporal():
                    do_retype_time = False
                if col == parent_codes_field and schema_updates[col] is not cls.PL_CODE_METADATA_SCHEMA[col]:
                    do_retype_parent_codes = False
            elif col in cls.PL_DATA_SCHEMA:
                read_schema[col] = cls.PL_DATA_SCHEMA[col]
            elif col in cls.PL_CODE_METADATA_SCHEMA:
                read_schema[col] = cls.PL_CODE_METADATA_SCHEMA[col]
            elif col in cls.PL_SUBJECT_SPLIT_SCHEMA:
                read_schema[col] = cls.PL_SUBJECT_SPLIT_SCHEMA[col]
            elif col in cls.PL_LABEL_SCHEMA:
                read_schema[col] = cls.PL_LABEL_SCHEMA[col]
            else:
                logger.warning(f"Column {col} not found in schema")

            if do_retype_time:
                time_schema[col] = read_schema.pop(col)
                read_schema[col] = pl.String
            elif do_retype_parent_codes:
                has_parent_codes = True
                read_schema[col] = pl.String

        try:
            df = pl.read_csv(
                StringIO(csv),
                schema_overrides={col: read_schema[col] for col in cols if col in read_schema},
            )
        except Exception as e:
            raise ValueError(f"Failed to read:\n{csv}\nUnder schema:\n{read_schema}") from e

        col_updates = {t: pl.col(t).str.strptime(dt, cls.CSV_TS_FORMAT) for t, dt in time_schema.items()}
        if has_parent_codes:
            col_updates[parent_codes_field] = (
                pl.col(parent_codes_field).str.split(", ").cast(pl.List(pl.String))
            )

        return df.with_columns(**col_updates).select(cols)

    @classmethod
    def from_yaml(cls, yaml: str | Path, **schema_overrides) -> "MEDSDataset":
        """Create a MEDSDataset from a YAML string or file on disk.

        Args:
            yaml: The YAML string or file path to load the dataset from. This file should contain a flat set
                of string keys that correspond to file-paths relative to the MEDS-Root, with values that are
                strings of the associated data in CSV format (JSON format for dataset metadata). Missing keys
                corresponding to mandatory files will be inferred if possible or raise an error if not.

        Raises:
            ValueError: If the YAML is not a valid MEDSDataset.
            FileNotFoundError: If the file path does not exist.

        Returns:
            The MEDSDataset object reflected in the YAML file. If no code metadata is specified, an empty code
            metadata dataframe will be created. If no subject splits are specified, `None` will be returned.
            If no dataset metadata is specified, a default dataset metadata object will be created.

        Examples:
            >>> from meds_testing_helpers.static_sample_data import SIMPLE_STATIC_SHARDED_BY_SPLIT
            >>> D = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)
            >>> print(D)
            MEDSDataset:
            dataset_metadata:
            data_shards:
              - train/0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[239684,239684,239684,239684,239684,...,1195293,1195293,1195293,1195293,1195293],[1195293]]
                time: [[null,null,1980-12-28 00:00:00.000000,2010-05-11 17:41:51.000000,2010-05-11 17:41:51.000000,...,2010-06-20 20:12:31.000000,2010-06-20 20:24:44.000000,2010-06-20 20:24:44.000000,2010-06-20 20:41:33.000000,2010-06-20 20:41:33.000000],[2010-06-20 20:50:04.000000]]
                code: [["EYE_COLOR//BROWN","HEIGHT","DOB","ADMISSION//CARDIAC","HR",...,"TEMP","HR","TEMP","HR","TEMP"],["DISCHARGE"]]
                numeric_value: [[null,175.27112,null,null,102.6,...,99.8,107.7,100,107.5,100.4],[null]]
              - train/1:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[68729,68729,68729,68729,68729,...,814703,814703,814703,814703,814703],[814703]]
                time: [[null,null,1978-03-09 00:00:00.000000,2010-05-26 02:30:56.000000,2010-05-26 02:30:56.000000,...,null,1976-03-28 00:00:00.000000,2010-02-05 05:55:39.000000,2010-02-05 05:55:39.000000,2010-02-05 05:55:39.000000],[2010-02-05 07:02:30.000000]]
                code: [["EYE_COLOR//HAZEL","HEIGHT","DOB","ADMISSION//PULMONARY","HR",...,"HEIGHT","DOB","ADMISSION//ORTHOPEDIC","HR","TEMP"],["DISCHARGE"]]
                numeric_value: [[null,160.39531,null,null,86,...,156.4856,null,null,170.2,100.1],[null]]
              - tuning/0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[754281,754281,754281,754281,754281,754281],[754281]]
                time: [[null,null,1988-12-19 00:00:00.000000,2010-01-03 06:27:59.000000,2010-01-03 06:27:59.000000,2010-01-03 06:27:59.000000],[2010-01-03 08:22:13.000000]]
                code: [["EYE_COLOR//BROWN","HEIGHT","DOB","ADMISSION//PULMONARY","HR","TEMP"],["DISCHARGE"]]
                numeric_value: [[null,166.22261,null,null,142,99.8],[null]]
              - held_out/0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[1500733,1500733,1500733,1500733,1500733,1500733,1500733,1500733,1500733,1500733],[1500733]]
                time: [[null,null,1986-07-20 00:00:00.000000,2010-06-03 14:54:38.000000,2010-06-03 14:54:38.000000,2010-06-03 14:54:38.000000,2010-06-03 15:39:49.000000,2010-06-03 15:39:49.000000,2010-06-03 16:20:49.000000,2010-06-03 16:20:49.000000],[2010-06-03 16:44:26.000000]]
                code: [["EYE_COLOR//BROWN","HEIGHT","DOB","ADMISSION//ORTHOPEDIC","HR","TEMP","HR","TEMP","HR","TEMP"],["DISCHARGE"]]
                numeric_value: [[null,158.60132,null,null,91.4,100,84.4,100.3,90.1,100.1],[null]]
            code_metadata:
              pyarrow.Table
              code: string
              description: string
              parent_codes: list<item: string>
                child 0, item: string
              ----
              code: [["EYE_COLOR//BLUE","EYE_COLOR//BROWN","EYE_COLOR//HAZEL","HR"],["TEMP"]]
              description: [["Blue Eyes. Less common than brown.","Brown Eyes. The most common eye color.","Hazel eyes. These are uncommon","Heart Rate"],["Body Temperature"]]
              parent_codes: [[null,null,null,["LOINC/8867-4"]],[["LOINC/8310-5"]]]
            subject_splits:
              pyarrow.Table
              subject_id: int64
              split: string
              ----
              subject_id: [[239684,1195293,68729,814703,754281],[1500733]]
              split: [["train","train","train","train","tuning"],["held_out"]]

            You can also read from a filepath directly:

            >>> import tempfile
            >>> yaml_lines = [
            ...    "data/train/0: |-2",
            ...    "  subject_id,time,code,numeric_value",
            ...    '  0,"1/1/2025, 12:00:00",A,',
            ...    "metadata/subject_splits.parquet: |-2",
            ...    "  subject_id,split",
            ...    "  0,train",
            ...    "metadata/dataset.json:",
            ...    "  dataset_name: test",
            ...    "  dataset_version: 0.0.1",
            ... ]
            >>> with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            ...     for line in yaml_lines:
            ...         _ = f.write(f"{line}\\n")
            ...     _ = f.flush()
            ...     D = MEDSDataset.from_yaml(f.name)
            ...     print(repr(D)) # doctest: +NORMALIZE_WHITESPACE
            MEDSDataset(data_shards={'train/0': {'subject_id': [0],
                                                 'time': [datetime.datetime(2025, 1, 1, 12, 0)],
                                                 'code': ['A'], 'numeric_value': [None]}},
                        dataset_metadata={'dataset_name': 'test',
                                          'dataset_version': '0.0.1'},
                        code_metadata={'code': [], 'description': [], 'parent_codes': []},
                        subject_splits={'subject_id': [0], 'split': ['train']})

            Though task labels are not formalized in MEDS (in terms of storage on disk; see
            https://github.com/Medical-Event-Data-Standard/meds/issues/75 for more information), you can also
            track a collection of sharded task labels by task name in this class:

            >>> from meds_testing_helpers.static_sample_data import SIMPLE_STATIC_SHARDED_BY_SPLIT_WITH_TASKS
            >>> D = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT_WITH_TASKS)
            >>> print(D)
            MEDSDataset:
            dataset_metadata:
            data_shards:
              - train/0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[239684,239684,239684,239684,239684,...,1195293,1195293,1195293,1195293,1195293],[1195293]]
                time: [[null,null,1980-12-28 00:00:00.000000,2010-05-11 17:41:51.000000,2010-05-11 17:41:51.000000,...,2010-06-20 20:12:31.000000,2010-06-20 20:24:44.000000,2010-06-20 20:24:44.000000,2010-06-20 20:41:33.000000,2010-06-20 20:41:33.000000],[2010-06-20 20:50:04.000000]]
                code: [["EYE_COLOR//BROWN","HEIGHT","DOB","ADMISSION//CARDIAC","HR",...,"TEMP","HR","TEMP","HR","TEMP"],["DISCHARGE"]]
                numeric_value: [[null,175.27112,null,null,102.6,...,99.8,107.7,100,107.5,100.4],[null]]
              - train/1:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[68729,68729,68729,68729,68729,...,814703,814703,814703,814703,814703],[814703]]
                time: [[null,null,1978-03-09 00:00:00.000000,2010-05-26 02:30:56.000000,2010-05-26 02:30:56.000000,...,null,1976-03-28 00:00:00.000000,2010-02-05 05:55:39.000000,2010-02-05 05:55:39.000000,2010-02-05 05:55:39.000000],[2010-02-05 07:02:30.000000]]
                code: [["EYE_COLOR//HAZEL","HEIGHT","DOB","ADMISSION//PULMONARY","HR",...,"HEIGHT","DOB","ADMISSION//ORTHOPEDIC","HR","TEMP"],["DISCHARGE"]]
                numeric_value: [[null,160.39531,null,null,86,...,156.4856,null,null,170.2,100.1],[null]]
              - tuning/0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[754281,754281,754281,754281,754281,754281],[754281]]
                time: [[null,null,1988-12-19 00:00:00.000000,2010-01-03 06:27:59.000000,2010-01-03 06:27:59.000000,2010-01-03 06:27:59.000000],[2010-01-03 08:22:13.000000]]
                code: [["EYE_COLOR//BROWN","HEIGHT","DOB","ADMISSION//PULMONARY","HR","TEMP"],["DISCHARGE"]]
                numeric_value: [[null,166.22261,null,null,142,99.8],[null]]
              - held_out/0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[1500733,1500733,1500733,1500733,1500733,1500733,1500733,1500733,1500733,1500733],[1500733]]
                time: [[null,null,1986-07-20 00:00:00.000000,2010-06-03 14:54:38.000000,2010-06-03 14:54:38.000000,2010-06-03 14:54:38.000000,2010-06-03 15:39:49.000000,2010-06-03 15:39:49.000000,2010-06-03 16:20:49.000000,2010-06-03 16:20:49.000000],[2010-06-03 16:44:26.000000]]
                code: [["EYE_COLOR//BROWN","HEIGHT","DOB","ADMISSION//ORTHOPEDIC","HR","TEMP","HR","TEMP","HR","TEMP"],["DISCHARGE"]]
                numeric_value: [[null,158.60132,null,null,91.4,100,84.4,100.3,90.1,100.1],[null]]
            code_metadata:
              pyarrow.Table
              code: string
              description: string
              parent_codes: list<item: string>
                child 0, item: string
              ----
              code: [["EYE_COLOR//BLUE","EYE_COLOR//BROWN","EYE_COLOR//HAZEL","HR"],["TEMP"]]
              description: [["Blue Eyes. Less common than brown.","Brown Eyes. The most common eye color.","Hazel eyes. These are uncommon","Heart Rate"],["Body Temperature"]]
              parent_codes: [[null,null,null,["LOINC/8867-4"]],[["LOINC/8310-5"]]]
            subject_splits:
              pyarrow.Table
              subject_id: int64
              split: string
              ----
              subject_id: [[239684,1195293,68729,814703,754281],[1500733]]
              split: [["train","train","train","train","tuning"],["held_out"]]
            task labels:
              * boolean_value_task:
                - labels_A.parquet:
                  pyarrow.Table
                  subject_id: int64
                  prediction_time: timestamp[us]
                  boolean_value: bool
                  integer_value: int64
                  float_value: double
                  categorical_value: string
                  ----
                  subject_id: [[239684,239684,239684,1195293,1195293,1195293,68729,68729,68729],[68729]]
                  prediction_time: [[2010-05-11 18:00:00.000000,2010-05-11 18:30:00.000000,2010-05-11 19:00:00.000000,2010-06-20 19:30:00.000000,2010-06-20 20:00:00.000000,2010-06-20 20:30:00.000000,2010-05-26 03:00:00.000000,2010-05-26 03:30:00.000000,2010-05-26 04:00:00.000000],[2010-05-26 04:30:00.000000]]
                  boolean_value: [[false,true,true,false,true,true,false,false,true],[true]]
                  integer_value: [[null,null,null,null,null,null,null,null,null],[null]]
                  float_value: [[null,null,null,null,null,null,null,null,null],[null]]
                  categorical_value: [[null,null,null,null,null,null,null,null,null],[null]]
                - labels_B.parquet:
                  pyarrow.Table
                  subject_id: int64
                  prediction_time: timestamp[us]
                  boolean_value: bool
                  integer_value: int64
                  float_value: double
                  categorical_value: string
                  ----
                  subject_id: [[814703,814703,814703,754281,754281,754281,754281,1500733,1500733,1500733],[1500733]]
                  prediction_time: [[2010-02-05 06:00:00.000000,2010-02-05 06:30:00.000000,2010-02-05 07:00:00.000000,2010-01-03 06:30:00.000000,2010-01-03 07:00:00.000000,2010-01-03 07:30:00.000000,2010-01-03 08:00:00.000000,2010-06-03 15:00:00.000000,2010-06-03 15:30:00.000000,2010-06-03 16:00:00.000000],[2010-06-03 16:30:00.000000]]
                  boolean_value: [[false,true,true,false,false,true,true,false,false,true],[true]]
                  integer_value: [[null,null,null,null,null,null,null,null,null,null],[null]]
                  float_value: [[null,null,null,null,null,null,null,null,null,null],[null]]
                  categorical_value: [[null,null,null,null,null,null,null,null,null,null],[null]]

            Errors are raised when the YAML is malformed or a non-existent path:

            >>> MEDSDataset.from_yaml(123)
            Traceback (most recent call last):
                ...
            ValueError: yaml must be a string or a file path; got <class 'int'>
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     MEDSDataset.from_yaml(Path(tmpdir) / "nonexistent.yaml")
            Traceback (most recent call last):
                ...
            FileNotFoundError: File not found: ...
            >>> MEDSDataset.from_yaml("foo: bar")
            Traceback (most recent call last):
                ...
            ValueError: Unrecognized key in YAML: foo. Must start with 'data/' or 'metadata/'.
            >>> MEDSDataset.from_yaml("metadata/codes.parquet: [1, 2]")
            Traceback (most recent call last):
                ...
            ValueError: Expected value for key metadata/codes.parquet to be a string, got <class 'list'>
            >>> MEDSDataset.from_yaml("metadata/foo: bar")
            Traceback (most recent call last):
                ...
            ValueError: Unrecognized key in YAML: metadata/foo
            >>> MEDSDataset.from_yaml("metadata/dataset.json: {dataset_name: test, dataset_version: 0.0.1}")
            Traceback (most recent call last):
                ...
            ValueError: No data shards found in YAML
            >>> MEDSDataset.from_yaml('metadata/dataset.json: "{dataset_name: test, dataset_version: 0.0.1}"')
            Traceback (most recent call last):
                ...
            ValueError: Expected value for key metadata/dataset.json to be a dict, got <class 'str'>
        """  # noqa: E501
        if isinstance(yaml, str) and yaml.endswith(".yaml"):
            logger.debug(f"Inferring yaml {yaml} is a file path as it ends with '.yaml'")
            yaml = Path(yaml)

        match yaml:
            case Path() if yaml.is_file():
                yaml = yaml.read_text().strip()
            case Path():
                raise FileNotFoundError(f"File not found: {yaml}")
            case str():
                yaml = yaml.strip()
            case _:
                raise ValueError(f"yaml must be a string or a file path; got {type(yaml)}")

        data = load_yaml(yaml, Loader=Loader)

        data_shards = {}
        code_metadata = None
        subject_splits = None
        dataset_metadata = DatasetMetadata()
        task_labels = None
        for key, value in data.items():
            key_parts = key.split("/")

            if key == cls.TASK_LABELS_SUBDIR:
                pass
            elif len(key_parts) < 2 or key_parts[0] not in {
                data_subdirectory,
                "metadata",
            }:
                raise ValueError(f"Unrecognized key in YAML: {key}. Must start with 'data/' or 'metadata/'.")

            if key in {dataset_metadata_filepath, cls.TASK_LABELS_SUBDIR}:
                if not isinstance(value, dict):
                    raise ValueError(f"Expected value for key {key} to be a dict, got {type(value)}")
            elif not isinstance(value, str):
                raise ValueError(f"Expected value for key {key} to be a string, got {type(value)}")

            root = key_parts[0]

            if root == data_subdirectory:
                rest = "/".join(key_parts[1:])
                data_shards[rest.replace(".parquet", "")] = cls.parse_csv(value, **schema_overrides)
            elif key == code_metadata_filepath:
                code_metadata = cls.parse_csv(value, **schema_overrides)
            elif key == subject_splits_filepath:
                subject_splits = cls.parse_csv(value, **schema_overrides)
            elif key == dataset_metadata_filepath:
                dataset_metadata = DatasetMetadata(**value, **schema_overrides)
            elif key == cls.TASK_LABELS_SUBDIR:
                task_labels = {
                    task_name: {
                        shard: cls.parse_csv(data, **schema_overrides) for shard, data in shards.items()
                    }
                    for task_name, shards in value.items()
                }
            else:
                raise ValueError(f"Unrecognized key in YAML: {key}")

        if len(data_shards) == 0:
            raise ValueError("No data shards found in YAML")

        return cls(
            data_shards=data_shards,
            dataset_metadata=dataset_metadata,
            code_metadata=code_metadata,
            subject_splits=subject_splits,
            task_labels=task_labels,
        )

    @staticmethod
    def _align_df(df: pl.DataFrame, schema: pa.Schema) -> pa.Table:
        return df.select(schema.names).to_arrow().cast(schema)

    @property
    def dataset_metadata_fp(self) -> Path | None:
        if self.root_dir is None:
            return None
        else:
            return self.root_dir / dataset_metadata_filepath

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        if self.root_dir is None:
            return self._dataset_metadata
        else:
            return DatasetMetadata(**json.loads(self.dataset_metadata_fp.read_text()))

    @dataset_metadata.setter
    def dataset_metadata(self, value: DatasetMetadata | None):
        self._dataset_metadata = value

    def _shard_name(self, data_fp: Path, root_dir: Path | None = None) -> str:
        if root_dir is None:
            root_dir = self.root_dir / data_subdirectory
        return data_fp.relative_to(root_dir).with_suffix("").as_posix()

    @property
    def shard_fps(self) -> list[Path] | None:
        if self.root_dir is None:
            return None
        else:
            return sorted((self.root_dir / data_subdirectory).rglob("*.parquet"))

    @property
    def _pl_shards(self) -> SHARDED_DF_T:
        if self._data_shards is None:
            return {self._shard_name(fp): pl.read_parquet(fp, use_pyarrow=True) for fp in self.shard_fps}
        else:
            return self._data_shards

    @property
    def data_shards(self) -> dict[str, pa.Table]:
        return {shard: self._align_df(df, data_schema()) for shard, df in self._pl_shards.items()}

    @data_shards.setter
    def data_shards(self, value: SHARDED_DF_T | None):
        self._data_shards = value

    @property
    def code_metadata_fp(self) -> Path | None:
        if self.root_dir is None:
            return None
        else:
            return self.root_dir / code_metadata_filepath

    @property
    def _pl_code_metadata(self) -> pl.DataFrame:
        if self._code_metadata is None:
            return pl.read_parquet(self.code_metadata_fp, use_pyarrow=True)
        else:
            return self._code_metadata

    @property
    def code_metadata(self) -> pa.Table:
        return self._align_df(self._pl_code_metadata, code_metadata_schema())

    @code_metadata.setter
    def code_metadata(self, value: pl.DataFrame | None):
        self._code_metadata = value

    @property
    def subject_splits_fp(self) -> Path | None:
        if self.root_dir is None:
            return None
        else:
            return self.root_dir / subject_splits_filepath

    @property
    def _pl_subject_splits(self) -> pl.DataFrame:
        if self.root_dir is None:
            return self._subject_splits

        if self.subject_splits_fp.exists():
            return pl.read_parquet(self.subject_splits_fp, use_pyarrow=True)
        else:
            return None

    @property
    def subject_splits(self) -> pa.Table | None:
        pl_subject_splits = self._pl_subject_splits
        if pl_subject_splits is None:
            return None
        return self._align_df(pl_subject_splits, subject_split_schema)

    @subject_splits.setter
    def subject_splits(self, value: pl.DataFrame | None):
        self._subject_splits = value

    @property
    def task_root_dir(self) -> Path | None:
        if self.root_dir is None:
            return None
        else:
            return self.root_dir / self.TASK_LABELS_SUBDIR

    @property
    def task_names_fp(self) -> Path | None:
        if self.task_root_dir is None:
            return None
        else:
            return self.task_root_dir / self.TASK_NAMES_FN

    @property
    def task_label_fps(self) -> dict[str, list[Path]] | None:
        if self.root_dir is None:
            return None

        if not self.task_root_dir.is_dir():
            return None
        else:
            out = {}
            task_names = json.loads(self.task_names_fp.read_text())
            for task in task_names:
                task_dir = self.task_root_dir / task
                out[task] = sorted(task_dir.rglob("*.parquet"))
            return out

    @property
    def _pl_task_labels(self) -> dict[str, SHARDED_DF_T] | None:
        if self.root_dir is None:
            return self._task_labels
        elif self.task_label_fps is None:
            return None
        else:
            out = {}
            for task, shard_fps in self.task_label_fps.items():
                out[task] = {
                    self._shard_name(fp, self.task_root_dir / task): pl.read_parquet(fp, use_pyarrow=True)
                    for fp in shard_fps
                }
            return out

    @property
    def task_labels(self) -> dict[str, pa.Table]:
        if self._pl_task_labels is None:
            return None

        return {
            task_name: {shard: self._align_df(df, label_schema) for shard, df in shards.items()}
            for task_name, shards in self._pl_task_labels.items()
        }

    @task_labels.setter
    def task_labels(self, value: dict[str, SHARDED_DF_T] | None):
        self._task_labels = value

    @property
    def task_names(self) -> list[str] | None:
        if self.task_labels is None:
            return None
        else:
            return list(self.task_labels.keys())

    def write(self, output_dir: Path) -> "MEDSDataset":
        data_dir = output_dir / data_subdirectory

        for shard, table in self.data_shards.items():
            fp = data_dir / f"{shard}.parquet"
            fp.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, fp)

        code_metadata_fp = output_dir / code_metadata_filepath
        code_metadata_fp.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self.code_metadata, code_metadata_fp)

        dataset_metadata_fp = output_dir / dataset_metadata_filepath
        dataset_metadata_fp.parent.mkdir(parents=True, exist_ok=True)
        dataset_metadata_fp.write_text(json.dumps(self.dataset_metadata))

        if self.subject_splits is not None:
            subject_splits_fp = output_dir / subject_splits_filepath
            subject_splits_fp.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(self.subject_splits, subject_splits_fp)

        if self.task_labels:
            task_labels_dir = output_dir / self.TASK_LABELS_SUBDIR
            task_labels_dir.mkdir(exist_ok=True, parents=True)

            task_names_fp = task_labels_dir / self.TASK_NAMES_FN
            task_names_fp.write_text(json.dumps(self.task_names))

            for task_name, shards in self.task_labels.items():
                for shard, table in shards.items():
                    fp = task_labels_dir / task_name / f"{shard}.parquet"
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table, fp)

        return MEDSDataset(root_dir=output_dir)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.root_dir is None:
            kwargs = {
                "data_shards": {k: v.to_dict(as_series=False) for k, v in self._pl_shards.items()},
                "dataset_metadata": self.dataset_metadata,
                "code_metadata": self._pl_code_metadata.to_dict(as_series=False),
            }
            if self.subject_splits is not None:
                kwargs["subject_splits"] = self._pl_subject_splits.to_dict(as_series=False)
            if self.task_labels is not None:
                kwargs["task_labels"] = {
                    task_name: {shard: df.to_dict(as_series=False) for shard, df in shards.items()}
                    for task_name, shards in self._pl_task_labels.items()
                }
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            return f"{cls_name}({kwargs_str})"
        else:
            return f"{cls_name}(root_dir={self.root_dir!r})"

    def __str__(self) -> str:
        lines = []
        lines.append(f"{self.__class__.__name__}:")
        if self.root_dir is not None:
            lines.append(f"stored in root_dir: {self.root_dir.resolve()!s}")
        lines.append("dataset_metadata:")
        for k, v in self.dataset_metadata.items():
            lines.append(f"  - {k}: {v}")
        lines.append("data_shards:")
        for shard, table in self.data_shards.items():
            lines.append(f"  - {shard}:")
            lines.append("    " + str(table).replace("\n", "\n    "))
        lines.append("code_metadata:")
        lines.append("  " + str(self.code_metadata).replace("\n", "\n  "))
        if self.subject_splits is None:
            lines.append("subject_splits: None")
        else:
            lines.append("subject_splits:")
            lines.append("  " + str(self.subject_splits).replace("\n", "\n  "))
        if self.task_labels is not None:
            lines.append("task labels:")
            for task_name, shards in self.task_labels.items():
                lines.append(f"  * {task_name}:")
                for shard, table in shards.items():
                    lines.append(f"    - {shard}:")
                    lines.append("      " + str(table).replace("\n", "\n      "))

        return "\n".join(lines)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MEDSDataset):
            return False

        if self.data_shards != other.data_shards:
            return False
        if self.dataset_metadata != other.dataset_metadata:
            return False
        if self.code_metadata != other.code_metadata:
            return False
        if self.subject_splits != other.subject_splits:
            return False
        return self.task_labels == other.task_labels
