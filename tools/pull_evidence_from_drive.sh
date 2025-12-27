#!/usr/bin/env bash
set -euo pipefail

REMOTE="gdrive:Arkiv/Juridik_och_identitetshandlingar"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$REPO_ROOT/evidence/land" "$REPO_ROOT/evidence/planning"

# Land/legal PDFs
rclone copy "$REMOTE" "$REPO_ROOT/evidence/land" \
  --filter "+ *LandsCommission*.pdf" \
  --filter "+ *LandsCommissino*.pdf" \
  --filter "+ *Indentrure*.pdf" \
  --filter "+ *Indentru*.pdf" \
  --filter "- *" \
  --progress

# Planning PDFs
rclone copy "$REMOTE" "$REPO_ROOT/evidence/planning" \
  --filter "+ *Description*.pdf" \
  --filter "- *" \
  --progress

echo "Done."
