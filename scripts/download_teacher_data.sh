#!/usr/bin/env bash
# Download the MarginMSE ensemble teacher scores (Hofstaetter et al., Zenodo 4068216).
#
# Usage:  bash scripts/download_teacher.sh
set -euo pipefail

# --- locate project root (parent of this script's dir) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$ROOT/data/teacher"
mkdir -p "$DEST_DIR"

# T2 ensemble teacher (matches the SPLADE distillation setting).
# For the single-teacher T1 variant, use bertbase_cat_msmarcopassage_train_scores_ids.tsv
FNAME="bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv"
URL="https://zenodo.org/records/4068216/files/${FNAME}?download=1"
MD5="4d99696386f96a7f1631076bcc53ac3c"
FILE="$DEST_DIR/$FNAME"

echo "target: $FILE"

# --- md5 helper (Linux md5sum, macOS md5 -q) ---
md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}';
  else md5 -q "$1"; fi
}

# --- skip if already present and valid ---
if [ -f "$FILE" ] && [ "$(md5_of "$FILE")" = "$MD5" ]; then
  echo "already downloaded and verified — nothing to do."
  exit 0
fi

# --- download with resume (-C -) + progress bar, following redirects (-L) ---
echo "downloading (~2.4 GB) ..."
curl -L -C - --fail --progress-bar -o "$FILE" "$URL"

# --- verify ---
echo "verifying md5 ..."
GOT="$(md5_of "$FILE")"
if [ "$GOT" != "$MD5" ]; then
  echo "MD5 MISMATCH: got $GOT expected $MD5" >&2
  echo "delete '$FILE' and re-run." >&2
  exit 1
fi
echo "md5 OK — teacher scores ready at $FILE"