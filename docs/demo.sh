#!/usr/bin/env bash
# clob — end-to-end demo. Run from repo root:
#     bash docs/demo.sh
#
# Or record an asciinema cast (needs `brew install asciinema` or pipx):
#     asciinema rec -c "bash docs/demo.sh" docs/demo.cast
#
# The script assumes you've already built — see README quickstart. If the
# binaries aren't there, it'll tell you what to run.

set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

# Prefer the asan build; fall back to release. Either is valid.
BIN_DIR=""
for candidate in build/Debug build/Release; do
  if [ -x "${candidate}/apps/matcher-cli" ]; then
    BIN_DIR="${candidate}"
    break
  fi
done
if [ -z "${BIN_DIR}" ]; then
  echo "No built binaries found. Build first:"
  echo "    conan install . -s build_type=Debug --build=missing"
  echo "    cmake --preset asan && cmake --build build/Debug --parallel"
  exit 1
fi

MATCHER="${BIN_DIR}/apps/matcher-cli"
REPLAY="${BIN_DIR}/apps/replay-cli"

WORK="$(mktemp -d -t clob_demo_XXXXXX)"
trap 'rm -rf "${WORK}"' EXIT

JOURNAL="${WORK}/session.journal.bin"
FILLS_LIVE="${WORK}/fills.live.bin"
FILLS_REPLAY="${WORK}/fills.replay.bin"

step() { printf "\n${CYAN}\$ %s${NC}\n" "$*"; }
note() { printf "${DIM}# %s${NC}\n" "$*"; }
ok()   { printf "${GREEN}✓ %s${NC}\n" "$*"; }

# ----- 1. Drive matcher-cli, journaling every accepted op ---------------------
note "Build a 2-sided book, then send an aggressive bid that walks the asks."
step "matcher-cli --journal=session.journal.bin <<EOF"
cat <<'EOF'
limit 1 ask 100 5
limit 2 ask 101 5
limit 3 bid 99  4
limit 4 bid 101 7
book
quit
EOF
echo

"${MATCHER}" --journal="${JOURNAL}" <<'EOF' | sed 's/^/    /'
limit 1 ask 100 5
limit 2 ask 101 5
limit 3 bid 99  4
limit 4 bid 101 7
book
quit
EOF

note "session.journal.bin captured every accepted op as a binary record."
step "ls -la ${JOURNAL}"
ls -la "${JOURNAL}" | awk '{print "    " $5, $NF}'
step "xxd ${JOURNAL} | head -4"
xxd "${JOURNAL}" | head -4 | sed 's/^/    /'

# ----- 2. Replay the journal through a fresh engine ---------------------------
note "Now replay it through a brand-new engine — output must be bit-identical."
step "replay-cli session.journal.bin fills.replay.bin"
"${REPLAY}" "${JOURNAL}" "${FILLS_REPLAY}" | sed 's/^/    /'

step "xxd ${FILLS_REPLAY}"
xxd "${FILLS_REPLAY}" | sed 's/^/    /'

# ----- 3. Re-replay to prove determinism --------------------------------------
note "Run replay AGAIN into a second file and diff them."
"${REPLAY}" "${JOURNAL}" "${FILLS_LIVE}" > /dev/null
step "cmp fills.replay.bin fills.replay2.bin"
if cmp -s "${FILLS_REPLAY}" "${FILLS_LIVE}"; then
  ok "byte-identical → replay is deterministic (ADR 0001)"
else
  echo "FAIL: replay produced different bytes on two runs"
  exit 1
fi

echo
ok "demo complete"
