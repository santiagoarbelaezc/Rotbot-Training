"""
Rotbot Training - Dataset Validator Module
Validates message formatting, role alternation, token statistics, style compliance, and structural integrity.
"""

import os
import re
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import Counter
import pandas as pd


# Unicode range regex to detect emojis reliably
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)


def has_emojis(text: str) -> bool:
    """Returns True if the text contains any emoji character."""
    if not text:
        return False
    return bool(EMOJI_PATTERN.search(text))


def extract_emojis(text: str) -> List[str]:
    """Returns a list of emojis found in the text."""
    if not text:
        return []
    return EMOJI_PATTERN.findall(text)


def estimate_tokens(text: str) -> int:
    """
    Estimates token count using tiktoken (cl100k_base) if available,
    falling back to character/word ratio heuristics (~4 chars/token).
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        words = len(text.split())
        chars = len(text)
        return max(words, int(chars / 4.0))


def compute_detailed_token_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comprehensive lexical and token statistics for a DataFrame with user_message and assistant_message.
    """
    if df.empty or 'user_message' not in df.columns or 'assistant_message' not in df.columns:
        return {}

    user_tokens = df['user_message'].astype(str).apply(estimate_tokens)
    asst_tokens = df['assistant_message'].astype(str).apply(estimate_tokens)
    total_tokens = user_tokens + asst_tokens

    user_words = df['user_message'].astype(str).apply(lambda x: len(x.split()))
    asst_words = df['assistant_message'].astype(str).apply(lambda x: len(x.split()))

    user_chars = df['user_message'].astype(str).apply(len)
    asst_chars = df['assistant_message'].astype(str).apply(len)

    stats = {
        "count": len(df),
        "user_tokens": {
            "mean": round(float(user_tokens.mean()), 1),
            "median": round(float(user_tokens.median()), 1),
            "min": int(user_tokens.min()),
            "max": int(user_tokens.max()),
            "p25": round(float(user_tokens.quantile(0.25)), 1),
            "p75": round(float(user_tokens.quantile(0.75)), 1),
            "sum": int(user_tokens.sum())
        },
        "assistant_tokens": {
            "mean": round(float(asst_tokens.mean()), 1),
            "median": round(float(asst_tokens.median()), 1),
            "min": int(asst_tokens.min()),
            "max": int(asst_tokens.max()),
            "p25": round(float(asst_tokens.quantile(0.25)), 1),
            "p75": round(float(asst_tokens.quantile(0.75)), 1),
            "sum": int(asst_tokens.sum())
        },
        "total_tokens": {
            "mean": round(float(total_tokens.mean()), 1),
            "median": round(float(total_tokens.median()), 1),
            "min": int(total_tokens.min()),
            "max": int(total_tokens.max()),
            "p25": round(float(total_tokens.quantile(0.25)), 1),
            "p75": round(float(total_tokens.quantile(0.75)), 1),
            "sum": int(total_tokens.sum())
        },
        "words": {
            "user_mean": round(float(user_words.mean()), 1),
            "asst_mean": round(float(asst_words.mean()), 1),
            "total_mean": round(float((user_words + asst_words).mean()), 1)
        },
        "chars": {
            "user_mean": round(float(user_chars.mean()), 1),
            "asst_mean": round(float(asst_chars.mean()), 1)
        },
        "asst_to_user_token_ratio": round(float(asst_tokens.mean() / max(user_tokens.mean(), 1)), 2)
    }
    return stats


def check_rotbot_style_compliance(df_or_records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Audits RotBot personality compliance:
    1. Presence of 'boss' in assistant responses.
    2. Strict absence of emojis.
    3. English response verification.
    """
    if isinstance(df_or_records, list):
        df = pd.DataFrame(df_or_records)
    else:
        df = df_or_records.copy()

    if df.empty:
        return {"total": 0, "boss_compliance_pct": 0.0, "emoji_free_pct": 100.0, "issues": []}

    target_col = "assistant_message"
    if target_col not in df.columns:
        for c in ["bot_response", "response", "output"]:
            if c in df.columns:
                target_col = c
                break

    total = len(df)
    boss_matches = 0
    emoji_violations = []
    missing_boss_indices = []

    for idx, text in enumerate(df[target_col].astype(str)):
        t_lower = text.lower()
        if re.search(r'\bboss\b', t_lower):
            boss_matches += 1
        else:
            missing_boss_indices.append(idx)

        if has_emojis(text):
            found = extract_emojis(text)
            emoji_violations.append((idx, found))

    boss_pct = round((boss_matches / max(total, 1)) * 100, 2)
    emoji_free_pct = round(((total - len(emoji_violations)) / max(total, 1)) * 100, 2)

    return {
        "total_evaluated": total,
        "boss_count": boss_matches,
        "boss_compliance_pct": boss_pct,
        "missing_boss_count": len(missing_boss_indices),
        "missing_boss_indices": missing_boss_indices[:10],
        "emoji_violations_count": len(emoji_violations),
        "emoji_violations": emoji_violations[:10],
        "emoji_free_pct": emoji_free_pct,
        "is_fully_compliant": (boss_pct >= 95.0 and len(emoji_violations) == 0)
    }


def validate_chatml_entry(entry: Dict[str, Any], index: int = 0) -> List[str]:
    """
    Validates a single ChatML format JSON object.
    """
    errors = []

    if "messages" in entry:
        messages = entry.get("messages")
        if not isinstance(messages, list):
            errors.append(f"Row {index}: 'messages' must be a list, got {type(messages).__name__}")
            return errors

        if len(messages) < 2:
            errors.append(f"Row {index}: Conversation must have at least 2 messages (found {len(messages)})")

        has_user = False
        has_assistant = False

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"Row {index}, msg {i}: Message must be a dictionary")
                continue

            role = msg.get("role")
            content = msg.get("content")

            if role not in ("system", "user", "assistant"):
                errors.append(f"Row {index}, msg {i}: Invalid role '{role}'. Expected 'system', 'user', or 'assistant'")

            if content is None or not isinstance(content, str) or content.strip() == "":
                errors.append(f"Row {index}, msg {i} ({role}): Message content is empty or not a string")

            if role == "user":
                has_user = True
            elif role == "assistant":
                has_assistant = True

        if not has_user:
            errors.append(f"Row {index}: Missing 'user' message")
        if not has_assistant:
            errors.append(f"Row {index}: Missing 'assistant' message")

    elif "contents" in entry:
        contents = entry.get("contents")
        if not isinstance(contents, list) or len(contents) < 2:
            errors.append(f"Row {index}: Gemini format 'contents' must be a list with at least 2 turns")
            return errors

        has_user = False
        has_model = False
        for i, turn in enumerate(contents):
            role = turn.get("role")
            parts = turn.get("parts", [])
            if role not in ("user", "model"):
                errors.append(f"Row {index}, turn {i}: Invalid role '{role}'. Expected 'user' or 'model'")
            if not parts or not isinstance(parts, list):
                errors.append(f"Row {index}, turn {i}: 'parts' must be a non-empty list")
            else:
                for p in parts:
                    if not p.get("text", "").strip():
                        errors.append(f"Row {index}, turn {i}: 'parts.text' is empty")

            if role == "user":
                has_user = True
            elif role == "model":
                has_model = True

        if not has_user or not has_model:
            errors.append(f"Row {index}: Gemini record must contain both 'user' and 'model' turns")

    else:
        errors.append(f"Row {index}: Unknown format. Must contain 'messages' (ChatML) or 'contents' (Gemini)")

    return errors


def validate_jsonl_dataset(jsonl_path: str) -> Dict[str, Any]:
    """
    Validates an entire .jsonl file, gathering statistics and checking for duplicates or format violations.
    """
    if not os.path.exists(jsonl_path):
        return {
            "valid": False,
            "total_records": 0,
            "errors": [f"File not found: {jsonl_path}"],
            "warnings": [],
            "stats": {}
        }

    errors = []
    warnings = []
    total_records = 0

    user_token_counts = []
    assistant_token_counts = []
    total_token_counts = []
    user_prompts_seen = Counter()
    categories_counter = Counter()
    level_counter = Counter()
    emoji_count = 0
    boss_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                warnings.append(f"Line {idx}: Empty blank line in JSONL file")
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as e:
                errors.append(f"Line {idx}: Invalid JSON syntax - {e}")
                continue

            total_records += 1
            row_errors = validate_chatml_entry(record, index=idx)
            errors.extend(row_errors)

            if "messages" in record:
                user_text = ""
                assistant_text = ""
                for msg in record["messages"]:
                    r = msg.get("role")
                    c = msg.get("content", "")
                    if r == "user":
                        user_text += c + " "
                    elif r == "assistant":
                        assistant_text += c + " "

                u_tok = estimate_tokens(user_text)
                a_tok = estimate_tokens(assistant_text)
                user_token_counts.append(u_tok)
                assistant_token_counts.append(a_tok)
                total_token_counts.append(u_tok + a_tok)

                cleaned_u = user_text.strip().lower()
                if cleaned_u:
                    user_prompts_seen[cleaned_u] += 1

                if has_emojis(assistant_text):
                    emoji_count += 1

                if re.search(r'\bboss\b', assistant_text.lower()):
                    boss_count += 1

                meta = record.get("_metadata", {})
                cat = meta.get("category", "Uncategorized")
                lvl = meta.get("level", "Unspecified")
                categories_counter[cat] += 1
                level_counter[lvl] += 1

            elif "contents" in record:
                u_text = " ".join([p.get("text", "") for t in record["contents"] if t.get("role") == "user" for p in t.get("parts", [])])
                m_text = " ".join([p.get("text", "") for t in record["contents"] if t.get("role") == "model" for p in t.get("parts", [])])
                u_tok = estimate_tokens(u_text)
                m_tok = estimate_tokens(m_text)
                user_token_counts.append(u_tok)
                assistant_token_counts.append(m_tok)
                total_token_counts.append(u_tok + m_tok)

                if has_emojis(m_text):
                    emoji_count += 1
                if re.search(r'\bboss\b', m_text.lower()):
                    boss_count += 1

    # Check duplicates
    duplicate_count = 0
    for prompt, count in user_prompts_seen.items():
        if count > 1:
            duplicate_count += 1
            if duplicate_count <= 5:
                warnings.append(f"Duplicate user prompt detected ({count} times): '{prompt[:60]}...'")

    if duplicate_count > 5:
        warnings.append(f"... and {duplicate_count - 5} more duplicate user prompts.")

    stats = {
        "total_records": total_records,
        "duplicate_prompts": duplicate_count,
        "categories": dict(categories_counter),
        "levels": dict(level_counter),
        "total_tokens_sum": sum(total_token_counts),
        "avg_tokens_per_example": round(sum(total_token_counts) / max(total_records, 1), 1),
        "max_tokens_example": max(total_token_counts) if total_token_counts else 0,
        "min_tokens_example": min(total_token_counts) if total_token_counts else 0,
        "avg_user_tokens": round(sum(user_token_counts) / max(total_records, 1), 1),
        "avg_assistant_tokens": round(sum(assistant_token_counts) / max(total_records, 1), 1),
        "boss_presence_pct": round((boss_count / max(total_records, 1)) * 100, 1),
        "emoji_violations_count": emoji_count
    }

    is_valid = (len(errors) == 0 and total_records > 0)

    return {
        "valid": is_valid,
        "total_records": total_records,
        "errors": errors,
        "warnings": warnings,
        "stats": stats
    }


def print_dataset_summary(validation_report: Dict[str, Any], title: str = "Dataset Validation Summary"):
    """
    Prints a formatted summary table of the validation report safely across all OS terminal encodings.
    """
    stats = validation_report.get("stats", {})
    valid = validation_report.get("valid", False)
    errors = validation_report.get("errors", [])
    warnings = validation_report.get("warnings", [])

    print("=" * 60)
    print(f"[*] {title.upper()}")
    print("=" * 60)
    status_str = "[PASS] VALID" if valid else "[FAIL] INVALID"
    print(f"Status: {status_str}")
    print(f"Total Examples: {stats.get('total_records', 0)}")
    print(f"Total Tokens (est.): {stats.get('total_tokens_sum', 0):,}")
    print(f"Avg Tokens/Example: {stats.get('avg_tokens_per_example', 0)} (User: {stats.get('avg_user_tokens', 0)} | Assistant: {stats.get('avg_assistant_tokens', 0)})")
    print(f"Min / Max Tokens: {stats.get('min_tokens_example', 0)} / {stats.get('max_tokens_example', 0)}")
    print(f"Duplicate Prompts: {stats.get('duplicate_prompts', 0)}")
    print(f"RotBot 'boss' Compliance: {stats.get('boss_presence_pct', 0)}%")
    print(f"Emoji Violations: {stats.get('emoji_violations_count', 0)}")

    if stats.get("categories"):
        print("\n[+] Category Distribution:")
        for cat, cnt in stats["categories"].items():
            print(f"  - {cat}: {cnt} examples ({cnt / max(stats.get('total_records', 1), 1) * 100:.1f}%)")

    if stats.get("levels"):
        print("\n[+] Level Distribution:")
        for lvl, cnt in stats["levels"].items():
            print(f"  - {lvl}: {cnt} examples ({cnt / max(stats.get('total_records', 1), 1) * 100:.1f}%)")

    if warnings:
        print(f"\n[!] Warnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more warnings.")

    if errors:
        print(f"\n[X] Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")
    print("=" * 60)
