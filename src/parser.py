"""
Rotbot Training - SQL & Data Parser Module
Provides functions to parse SQL dumps/INSERTs, CSVs, and structured data into ChatML/Gemini JSONL formats.
"""

import os
import re
import glob
import json
import random
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd

DEFAULT_SYSTEM_PROMPT = (
    "You are Rotbot, an English teacher with a personality that is friendly, sarcastic, and clever. "
    "You always call the user 'boss'. In this interaction, your goal is to correct common grammar mistakes, "
    "explaining the rule clearly with a touch of humor, without being mean. You never use emojis. You always respond in English."
)


def _split_sql_values_row(row_str: str) -> List[str]:
    """
    Parses a single SQL values row (e.g., "1, 'Hello', 'World''s end', 'grammar'")
    respecting escaped quotes and commas inside string literals.
    """
    values = []
    current = []
    in_quote = False
    quote_char = None
    i = 0
    length = len(row_str)

    while i < length:
        ch = row_str[i]

        if not in_quote:
            if ch in ("'", '"', '`'):
                in_quote = True
                quote_char = ch
            elif ch == ',':
                val = "".join(current).strip()
                values.append(val)
                current = []
                i += 1
                continue
            else:
                current.append(ch)
        else:
            if ch == '\\' and i + 1 < length:
                next_ch = row_str[i + 1]
                if next_ch in ("'", '"', '\\', 'n', 'r', 't'):
                    if next_ch == 'n':
                        current.append('\n')
                    elif next_ch == 't':
                        current.append('\t')
                    elif next_ch == 'r':
                        current.append('\r')
                    else:
                        current.append(next_ch)
                    i += 2
                    continue
                else:
                    current.append(ch)
            elif ch == quote_char:
                # Check for SQL doubled quotes '' or ""
                if i + 1 < length and row_str[i + 1] == quote_char:
                    current.append(quote_char)
                    i += 2
                    continue
                else:
                    in_quote = False
                    quote_char = None
            else:
                current.append(ch)

        i += 1

    if current or row_str.endswith(','):
        values.append("".join(current).strip())

    # Clean up outer quotes and NULLs
    cleaned_values = []
    for v in values:
        v_clean = v.strip()
        if (v_clean.startswith("'") and v_clean.endswith("'")) or (v_clean.startswith('"') and v_clean.endswith('"')):
            v_clean = v_clean[1:-1]
        elif v_clean.upper() == 'NULL':
            v_clean = None
        cleaned_values.append(v_clean)

    return cleaned_values


def find_default_sql_file(base_dir: Optional[str] = None) -> Optional[str]:
    """
    Finds the primary raw SQL dataset in data/raw/ prioritizing rotbot_training_dataset.sql.
    """
    search_dir = base_dir or os.getcwd()
    raw_dir = os.path.join(search_dir, "data", "raw")
    if not os.path.exists(raw_dir):
        raw_dir = search_dir

    preferred = os.path.join(raw_dir, "rotbot_training_dataset.sql")
    if os.path.exists(preferred):
        return preferred

    sql_files = glob.glob(os.path.join(raw_dir, "*.sql"))
    if sql_files:
        return sql_files[0]
    return None


def parse_sql_inserts(sql_content_or_path: str) -> List[Dict[str, Any]]:
    """
    Extracts column names and rows from SQL INSERT INTO statements robustly.
    
    Args:
        sql_content_or_path: String containing raw SQL statements or path to a .sql file.
        
    Returns:
        List of dictionaries representing extracted records.
    """
    if os.path.exists(sql_content_or_path):
        with open(sql_content_or_path, 'r', encoding='utf-8', errors='replace') as f:
            sql_text = f.read()
    else:
        sql_text = sql_content_or_path

    # Clean line comments starting with -- or /* at line start (avoid stripping -- inside string literals)
    cleaned_lines = []
    for line in sql_text.splitlines():
        st = line.strip()
        if st.startswith('--') or st.startswith('/*'):
            continue
        cleaned_lines.append(line)
    sql_text = '\n'.join(cleaned_lines)

    records = []

    # Match each INSERT INTO statement
    insert_pattern = re.compile(
        r'INSERT\s+INTO\s+[`"]?(\w+)[`"]?\s*(?:\(([^)]+)\))?\s*VALUES\s*(.+?);',
        re.IGNORECASE | re.DOTALL
    )

    for match in insert_pattern.finditer(sql_text):
        raw_cols = match.group(2)
        raw_values_block = match.group(3).strip()

        cols = []
        if raw_cols:
            cols = [c.strip().strip('`"\' ') for c in raw_cols.split(',')]

        # Tokenize tuple rows with full quote-aware state machine
        i = 0
        block_len = len(raw_values_block)

        while i < block_len:
            # Advance to opening paren of tuple
            while i < block_len and raw_values_block[i] != '(':
                i += 1
            if i >= block_len:
                break

            start = i
            in_q = False
            q_char = None
            end = -1
            j = start + 1

            while j < block_len:
                ch = raw_values_block[j]
                if not in_q:
                    if ch in ("'", '"', '`'):
                        in_q = True
                        q_char = ch
                    elif ch == ')':
                        end = j
                        break
                else:
                    if ch == '\\' and j + 1 < block_len:
                        j += 2
                        continue
                    elif ch == q_char:
                        if j + 1 < block_len and raw_values_block[j + 1] == q_char:
                            j += 2
                            continue
                        in_q = False
                        q_char = None
                j += 1

            if end != -1:
                row_content = raw_values_block[start + 1:end].strip()
                parsed_vals = _split_sql_values_row(row_content)

                if cols and len(cols) == len(parsed_vals):
                    row_dict = dict(zip(cols, parsed_vals))
                else:
                    row_dict = {f"col_{idx}": val for idx, val in enumerate(parsed_vals)}

                records.append(row_dict)
                i = end + 1
            else:
                break

    return records


def sql_to_dataframe(sql_content_or_path: str) -> pd.DataFrame:
    """
    Parses SQL inserts and converts them into a standardized pandas DataFrame.
    Automatically standardizes column names.
    """
    records = parse_sql_inserts(sql_content_or_path)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Column mapping heuristics
    col_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in ('user_input', 'user_message', 'user_prompt', 'prompt', 'input', 'student_text', 'user'):
            col_mapping[col] = 'user_message'
        elif col_lower in ('bot_response', 'assistant_response', 'response', 'completion', 'output', 'coach_response', 'assistant'):
            col_mapping[col] = 'assistant_message'
        elif col_lower in ('category', 'topic', 'tag', 'type', 'intent'):
            col_mapping[col] = 'category'
        elif col_lower in ('level', 'difficulty_level', 'difficulty'):
            col_mapping[col] = 'level'
        elif col_lower in ('system_prompt', 'system', 'instruction'):
            col_mapping[col] = 'system_prompt'
        elif col_lower in ('error_targeted', 'explanation', 'notes', 'feedback'):
            col_mapping[col] = 'notes'

    df = df.rename(columns=col_mapping)
    return df


def build_chatml_record(
    user_message: str,
    assistant_message: str,
    system_prompt: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Builds a standard OpenAI / ChatML format training record.
    """
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": str(sys_prompt).strip()},
        {"role": "user", "content": str(user_message).strip()},
        {"role": "assistant", "content": str(assistant_message).strip()}
    ]
    record: Dict[str, Any] = {"messages": messages}
    if metadata:
        record["_metadata"] = metadata
    return record


def build_gemini_tuning_record(
    user_message: str,
    assistant_message: str,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Builds a record in the Google AI Studio / Gemini fine-tuning JSON format (contents/parts).
    """
    sys_instruction = system_prompt or DEFAULT_SYSTEM_PROMPT
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": str(sys_instruction).strip()}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": str(user_message).strip()}]
            },
            {
                "role": "model",
                "parts": [{"text": str(assistant_message).strip()}]
            }
        ]
    }


def sql_to_jsonl(
    sql_path: str,
    output_train_path: str,
    output_val_path: str,
    val_ratio: float = 0.2,
    system_prompt: Optional[str] = None,
    format_type: str = "chatml",
    seed: int = 42
) -> Tuple[int, int]:
    """
    Complete pipeline to parse SQL, convert to training format, split, and save to train.jsonl and val.jsonl.
    """
    df = sql_to_dataframe(sql_path)
    if df.empty:
        raise ValueError(f"No records found in {sql_path}")

    if 'user_message' not in df.columns or 'assistant_message' not in df.columns:
        raise KeyError(
            f"Columns 'user_message' and 'assistant_message' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Clean data (drop nulls, strip strings)
    df = df.dropna(subset=['user_message', 'assistant_message']).copy()
    df['user_message'] = df['user_message'].astype(str).str.strip()
    df['assistant_message'] = df['assistant_message'].astype(str).str.strip()
    df = df[(df['user_message'] != '') & (df['assistant_message'] != '')]

    records = []
    for _, row in df.iterrows():
        user_msg = row['user_message']
        asst_msg = row['assistant_message']
        # If the row has its own system_prompt, use it; otherwise fallback to custom or DEFAULT
        sys_msg = row.get('system_prompt')
        if not sys_msg or pd.isna(sys_msg) or str(sys_msg).strip() == '':
            sys_msg = system_prompt or DEFAULT_SYSTEM_PROMPT

        meta = {}
        if 'category' in row and pd.notna(row['category']):
            meta['category'] = str(row['category'])
        if 'level' in row and pd.notna(row['level']):
            meta['level'] = str(row['level'])
        if 'notes' in row and pd.notna(row['notes']):
            meta['error_targeted'] = str(row['notes'])
        if 'id' in row and pd.notna(row['id']):
            meta['id'] = str(row['id'])

        if format_type == "gemini":
            rec = build_gemini_tuning_record(user_msg, asst_msg, sys_msg)
        else:
            rec = build_chatml_record(user_msg, asst_msg, sys_msg, metadata=meta if meta else None)

        records.append(rec)

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(records)

    # Split train / val
    num_total = len(records)
    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val

    train_data = records[:num_train]
    val_data = records[num_train:]

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(output_train_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_val_path)), exist_ok=True)

    with open(output_train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(output_val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return len(train_data), len(val_data)
