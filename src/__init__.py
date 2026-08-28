"""
Rotbot Training Source Package
Contains parsers, validators, and utility modules for training data preparation and model fine-tuning.
"""

from .parser import (
    parse_sql_inserts,
    build_chatml_record,
    sql_to_dataframe,
    sql_to_jsonl,
    DEFAULT_SYSTEM_PROMPT
)
from .validator import (
    validate_chatml_entry,
    validate_jsonl_dataset,
    print_dataset_summary
)

__all__ = [
    "parse_sql_inserts",
    "build_chatml_record",
    "sql_to_dataframe",
    "sql_to_jsonl",
    "DEFAULT_SYSTEM_PROMPT",
    "validate_chatml_entry",
    "validate_jsonl_dataset",
    "print_dataset_summary"
]
