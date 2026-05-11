from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

KEY_PATTERN = re.compile(r"PARTS?\s+LIST|BILL OF MATERIALS|BOM|PART NUMBER|ITEM", re.IGNORECASE)
STOP_PATTERN = re.compile(r"THIS DOCUMENT BELONGS|REVISION HISTORY|DO NOT SCALE|IF IN DOUBT ASK", re.IGNORECASE)
HEADER_VARIANTS = {
    "DESCRIPTION",
    "PART",
    "NUMBER",
    "PART NUMBER",
    "QTY",
    "ITEM",
    "DESCRIPTIONPART",
    "NUMBERQTYITEM",
    "DESCRIPTIONPART NUMBERQTYITEM",
    "DESCRIPTIONPART NUMBERQTY ITEM",
}


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_row_line(value: str) -> str:
    text = clean_text(value)
    for token in (
        "DESCRIPTIONPART NUMBERQTYITEM",
        "DESCRIPTIONPART NUMBERQTY ITEM",
        "DESCRIPTIONPART",
        "NUMBERQTYITEM",
    ):
        text = text.replace(token, "")
    return text.strip(" -")


def looks_like_part_number(token: str) -> bool:
    text = str(token or "").upper().rstrip(".")
    return bool(
        text
        and " " not in text
        and re.fullmatch(r"[A-Z0-9._/\\-]+", text)
        and re.search(r"\d", text)
        and 4 <= len(text) <= 16
    )


def extract_part_candidate(prefix: str) -> tuple[str, str] | None:
    raw = str(prefix or "").rstrip(" -")
    candidates: list[tuple[int, str, str]] = []
    for index, char in enumerate(raw):
        next_char = raw[index + 1] if index + 1 < len(raw) else ""
        prev_char = raw[index - 1] if index > 0 else " "
        if not (char.isdigit() or (char.isupper() and next_char.isdigit())):
            continue
        token = raw[index:].strip()
        description = raw[:index].strip(" -")
        if len(description) < 3 or not looks_like_part_number(token):
            continue
        if len(re.findall(r"[A-Za-z]", description)) < 2:
            continue
        score = 0
        if re.match(r"^[A-Z]{1,3}\d", token):
            score += 10
        if re.match(r"^\d", token):
            score += 9
        if 5 <= len(token) <= 10:
            score += 5
        if prev_char.isdigit():
            score -= 20
        else:
            score += 7
        if char.isdigit() and prev_char.isalpha():
            score += 8
        if char.isupper() and next_char.isdigit():
            score += 5
        if char.isupper() and prev_char.isupper():
            score -= 4
        if description[-1].isdigit():
            score -= 6
        if token.upper().startswith(("EE", "EL", "D", "H", "K", "P", "S")):
            score += 3
        candidates.append((score, description, token.rstrip(".")))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidates[0][1], candidates[0][2]


def parse_compact_row(line: str) -> dict[str, str] | None:
    normalized = normalize_row_line(line)
    if not normalized or not any(char.isdigit() for char in normalized):
        return None
    candidates: list[tuple[int, dict[str, str]]] = []
    for item_len in (2, 1):
        for qty_len in (2, 1):
            if len(normalized) <= item_len + qty_len + 4:
                continue
            item = normalized[-item_len:]
            qty = normalized[-(qty_len + item_len):-item_len]
            if not (item.isdigit() and qty.isdigit()):
                continue
            if not (0 < int(item) <= 250 and 0 < int(qty) <= 250):
                continue
            prefix = normalized[:-(qty_len + item_len)]
            candidate = extract_part_candidate(prefix)
            if not candidate:
                continue
            description, part_number = candidate
            score = (2 if item_len == 1 else 0) + (1 if qty_len == 1 else 0)
            candidates.append(
                (
                    score,
                    {
                        "item": item,
                        "qty_per_assembly": qty,
                        "part": description,
                        "part_number": part_number,
                        "raw_line": normalized,
                    },
                )
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def extract_rows_from_page(text: str, manual_name: str, page_number: int) -> list[dict[str, str | int]]:
    lines = [clean_text(line) for line in str(text or "").splitlines() if clean_text(line)]
    rows: list[dict[str, str | int]] = []
    seen: set[tuple[str, int, str, str, str, str]] = set()
    started = False
    for line in lines:
        upper = line.upper()
        if not started and KEY_PATTERN.search(upper):
            started = True
            continue
        if not started:
            continue
        if STOP_PATTERN.search(upper):
            break
        if upper in HEADER_VARIANTS:
            continue
        if re.fullmatch(r"[A-H]\s+[A-H]", upper):
            continue
        parsed = parse_compact_row(line)
        if not parsed:
            continue
        row = {
            "manual": manual_name,
            "page": int(page_number),
            "item": parsed["item"],
            "qty_per_assembly": parsed["qty_per_assembly"],
            "part": parsed["part"],
            "part_number": parsed["part_number"],
            "raw_line": parsed["raw_line"],
        }
        key = (
            row["manual"],
            row["page"],
            row["item"],
            row["part"],
            row["part_number"],
            row["qty_per_assembly"],
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def build_index(manuals_dir: Path) -> dict[str, object]:
    manuals: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    pdf_files = sorted(manuals_dir.glob("*.pdf"))
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            continue
        page_count = len(reader.pages)
        bom_pages: list[int] = []
        manual_rows: list[dict[str, object]] = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if not KEY_PATTERN.search(text):
                continue
            bom_pages.append(page_index)
            manual_rows.extend(extract_rows_from_page(text, pdf_path.name, page_index))
        rows.extend(manual_rows)
        manuals.append(
            {
                "name": pdf_path.name,
                "file_name": pdf_path.name,
                "path": str(pdf_path),
                "pages": page_count,
                "bom_pages": bom_pages,
                "row_count": len(manual_rows),
            }
        )
    return {
        "ok": True,
        "manuals": manuals,
        "rows": rows,
        "totals": {
            "manual_count": len(manuals),
            "row_count": len(rows),
        },
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "message": "Manuals directory is required."}))
        return 1
    manuals_dir = Path(sys.argv[1]).expanduser().resolve()
    if not manuals_dir.exists():
        print(json.dumps({"ok": True, "manuals": [], "rows": [], "totals": {"manual_count": 0, "row_count": 0}}))
        return 0
    payload = build_index(manuals_dir)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
