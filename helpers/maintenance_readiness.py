from typing import Dict, List


def safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def parse_parts_list(raw_text: str) -> List[str]:
    s = safe_str(raw_text)
    if not s:
        return []
    out = []
    for token in s.replace("\n", ",").replace(";", ",").replace("|", ",").split(","):
        t = token.strip()
        if t:
            out.append(t)
    uniq = []
    seen = set()
    for p in out:
        lk = p.lower()
        if lk not in seen:
            uniq.append(p)
            seen.add(lk)
    return uniq


def is_parts_conditional(task_text: str, procedure_text: str = "", prep_text: str = "") -> bool:
    txt = " ".join([safe_str(task_text).lower(), safe_str(procedure_text).lower(), safe_str(prep_text).lower()])
    markers = [
        "if needed",
        "if required",
        "as needed",
        "replace if",
        "if worn",
        "if eroded",
        "if damaged",
        "inspect/replace",
        "inspect and replace if",
    ]
    return any(m in txt for m in markers)


def compute_readiness(
    *,
    task_row,
    package_row: dict,
    stock_qty_map: Dict[str, float],
) -> dict:
    req_parts = parse_parts_list(task_row.get("Required_Parts", ""))
    prep = safe_str(package_row.get("Preparation_Checklist", ""))
    safety = safe_str(package_row.get("Safety_Protocol", ""))
    proc = safe_str(package_row.get("Procedure_Steps", ""))
    stop = safe_str(package_row.get("Draw_Stop_Plan", ""))
    completion = safe_str(package_row.get("Completion_Criteria", ""))

    conditional_parts = is_parts_conditional(
        task_text=task_row.get("Task", ""),
        procedure_text=proc,
        prep_text=prep,
    )
    missing_parts = [p for p in req_parts if float(stock_qty_map.get(p.lower(), 0.0)) <= 0.0]

    blockers = []
    warnings = []
    if missing_parts:
        if conditional_parts:
            warnings.append("Conditional parts check: some parts currently missing")
        else:
            blockers.append("Missing required parts")
    if not safety:
        warnings.append("Safety protocol not defined in work package")
    if not proc:
        warnings.append("Procedure steps not defined in work package")
    if not prep:
        warnings.append("Preparation checklist not defined in work package")
    if not stop:
        warnings.append("Draw stop plan not defined in work package")
    if not completion:
        warnings.append("Completion criteria not defined")

    score = 100
    if blockers:
        score -= 40
    score -= min(30, 10 * len([w for w in warnings if "not defined" in w]))
    if score < 0:
        score = 0

    ready = len(blockers) == 0
    if ready and score >= 80:
        readiness_label = "READY"
    elif ready:
        readiness_label = "READY (with warnings)"
    else:
        readiness_label = "BLOCKED"

    return {
        "ready_to_start": ready,
        "readiness_label": readiness_label,
        "score": int(score),
        "conditional_parts": bool(conditional_parts),
        "required_parts": req_parts,
        "missing_parts": missing_parts,
        "blockers": blockers,
        "warnings": warnings,
    }

