#!/usr/bin/env bash
# =================================================
# After-Done Hook — FULL CSV → TXT SNAPSHOT + PRINT (macOS)
# =================================================

set -euo pipefail

TARGET_CSV="${1:-}"
PREFORM_LEN_CM="${2:-0}"
DONE_DESC="${3:-}"

HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HOOK_DIR}/.." && pwd)"
DATASET_DIR="${PROJECT_ROOT}/data_set_csv"
OUT_DIR="${HOOK_DIR}/done_csv_snapshots"

mkdir -p "${OUT_DIR}"

CSV_PATH="${DATASET_DIR}/${TARGET_CSV}"

NOW="$(date '+%Y-%m-%d %H:%M:%S')"
SAFE_NAME="${TARGET_CSV%.csv}"
OUT_FILE="${OUT_DIR}/${SAFE_NAME}.txt"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "ERROR: CSV not found: ${CSV_PATH}" >> "${OUT_DIR}/errors.txt"
  exit 1
fi

DONE_DESC_CLEAN="$(echo "${DONE_DESC}" | tr '\r\n' ' ' | sed 's/  */ /g')"

{
  echo "============================================================"
  echo "DRAW DATASET — TEXT SNAPSHOT"
  echo "============================================================"
  echo "Time: ${NOW}"
  echo "Source CSV: ${TARGET_CSV}"
  echo "Preform Length After Draw (cm): ${PREFORM_LEN_CM}"
  echo "Done Description: ${DONE_DESC_CLEAN}"
  echo "------------------------------------------------------------"
  echo ""
  echo "CSV CONTENT (raw):"
  echo ""
  cat "${CSV_PATH}"
  echo ""
  echo "============================================================"
} > "${OUT_FILE}"

echo "TXT snapshot created: ${OUT_FILE}"

# =================================================
# macOS PRINT
# =================================================
lp "${OUT_FILE}" && echo "Sent to printer ✔"