"""
Rotbot Training Source Package
Contains parsers, validators, and utility modules for training data preparation and model fine-tuning.
"""

from .parser import (
    parse_sql_inserts,
    build_chatml_record,
    build_gemini_tuning_record,
    sql_to_dataframe,
    sql_to_jsonl,
    stratified_split,
    find_default_sql_file,
    DEFAULT_SYSTEM_PROMPT
)
from .validator import (
    validate_chatml_entry,
    validate_jsonl_dataset,
    print_dataset_summary,
    compute_detailed_token_stats,
    check_rotbot_style_compliance,
    has_emojis,
    estimate_tokens
)

__all__ = [
    "parse_sql_inserts",
    "build_chatml_record",
    "build_gemini_tuning_record",
    "sql_to_dataframe",
    "sql_to_jsonl",
    "stratified_split",
    "find_default_sql_file",
    "DEFAULT_SYSTEM_PROMPT",
    "validate_chatml_entry",
    "validate_jsonl_dataset",
    "print_dataset_summary",
    "compute_detailed_token_stats",
    "check_rotbot_style_compliance",
    "has_emojis",
    "estimate_tokens"
]
