#!/usr/bin/env bash
# Download MS MARCO passage data (collection + train/dev/DL19) for ir_datasets.
#
# Usage:  bash scripts/download_msmarco.sh
set -euo pipefail

# locate project root (parent of this script's dir)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
export IR_DATASETS_HOME="$ROOT/data/ir_datasets"
CACHE_DIR="$IR_DATASETS_HOME/downloads"
mkdir -p "$CACHE_DIR"
echo "IR_DATASETS_HOME = $IR_DATASETS_HOME"

# collection tarball
# TARBALL name is ir_datasets' cache key for this URL (printed in its log as the
# "symlink here to avoid downloading" path). If a future ir_datasets version
# prints a different hash, update TARBALL to match that path.
TAR_URL="https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
TARBALL="$CACHE_DIR/31644046b18952c1386cd4564ba2ae69"

if [ -f "$TARBALL" ]; then
  echo "collection tarball already present — skipping download"
else
  echo "downloading collection tarball (~1 GB) with curl ..."
  curl -L -C - --fail --progress-bar -o "$TARBALL.part" "$TAR_URL"
  mv "$TARBALL.part" "$TARBALL"
  echo "download complete."
fi

# use the project's uv env if available, else plain python 
cd "$ROOT"
if command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
else
  PY=(python3)
fi

# ir_datasets now finds the pre-placed tarball, verifies it, extracts,
# and builds the docstore (random-access .get(pid) lookup)
"${PY[@]}" - <<'PY'
import ir_datasets

c = ir_datasets.load("msmarco-passage")
print("collection docs (metadata):", c.docs_count(), flush=True)

print("extracting + building docstore (first run only) ...", flush=True)
store = c.docs_store()
s = store.get("0")                      # triggers extraction + docstore build
print("docstore ready:", s.doc_id, s.text[:60], flush=True)

for name, dsid in [
    ("train", "msmarco-passage/train"),
    ("dev",   "msmarco-passage/dev/small"),
    ("dl19",  "msmarco-passage/trec-dl-2019/judged"),
]:
    d = ir_datasets.load(dsid)
    print(f"{name:5s} queries:", sum(1 for _ in d.queries_iter()), flush=True)

print("done.", flush=True)
PY

echo "MS MARCO ready under $IR_DATASETS_HOME"