"""
Rotbot Training - Dataset Validator Module
Validates message formatting, role alternation, token statistics, and structural integrity of JSONL datasets.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter


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
        # Fallback estimation heuristic
        words = len(text.split())
        chars = len(text)
        return max(words, int(chars / 4.0))


def validate_chatml_entry(entry: Dict[str, Any], index: int = 0) -> List[str]:
    """
    Validates a single ChatML format JSON object.
    
    Returns:
        List of error strings found in this entry.
    """
    errors = []

    # Check for ChatML format
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

    # Check for Gemini format
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
    
    Returns:
        Dictionary with status, statistics, errors, and warnings.
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

            # Gather token and content statistics
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

                meta = record.get("_metadata", {})
                cat = meta.get("category", "Uncategorized")
                categories_counter[cat] += 1

            elif "contents" in record:
                u_text = " ".join([p.get("text", "") for t in record["contents"] if t.get("role") == "user" for p in t.get("parts", [])])
                m_text = " ".join([p.get("text", "") for t in record["contents"] if t.get("role") == "model" for p in t.get("parts", [])])
                u_tok = estimate_tokens(u_text)
                m_tok = estimate_tokens(m_text)
                user_token_counts.append(u_tok)
                assistant_token_counts.append(m_tok)
                total_token_counts.append(u_tok + m_tok)

    # Check duplicates
    duplicate_count = 0
    for prompt, count in user_prompts_seen.items():
        if count > 1:
            duplicate_count += 1
            if duplicate_count <= 5:
                warnings.append(f"Duplicate user prompt detected ({count} times): '{prompt[:60]}...'")

    if duplicate_count > 5:
        warnings.append(f"... and {duplicate_count - 5} more duplicate user prompts.")

    # Calculate statistics
    stats = {
        "total_records": total_records,
        "duplicate_prompts": duplicate_count,
        "categories": dict(categories_counter),
        "total_tokens_sum": sum(total_token_counts),
        "avg_tokens_per_example": round(sum(total_token_counts) / max(total_records, 1), 1),
        "max_tokens_example": max(total_token_counts) if total_token_counts else 0,
        "min_tokens_example": min(total_token_counts) if total_token_counts else 0,
        "avg_user_tokens": round(sum(user_token_counts) / max(total_records, 1), 1),
        "avg_assistant_tokens": round(sum(assistant_token_counts) / max(total_records, 1), 1)
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

    if stats.get("categories"):
        print("\n[+] Category Distribution:")
        for cat, cnt in stats["categories"].items():
            print(f"  - {cat}: {cnt} examples ({cnt / max(stats.get('total_records', 1), 1) * 100:.1f}%)")

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
