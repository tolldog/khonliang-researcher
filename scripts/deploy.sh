#!/usr/bin/env bash
#
# Deploy khonliang-researcher from its GitHub source clone into its service
# venv, then restart its agents on the bus.
#
# Researcher ships TWO bus agents from one venv/source tree:
#   - researcher-primary  (python -m researcher.agent)
#   - librarian-primary   (python -m researcher.librarian_agent)
# Both are reinstalled together and restarted together.
#
# Deploy model (per operator, mirrors scripts/deploy.sh in the sibling agents):
# pull the source clone from GitHub, reinstall the package into the (copied,
# non-editable) service venv, and ONLY if pip requirements changed also sync
# dependencies -- all as the venv-owning user. Restart is bus-managed (no
# systemd).
#
# The service venv has no setuptools, so `pip install <sourcetree>` builds via
# pip's build isolation, which fetches setuptools from PyPI. The box must be
# online; the script fails loudly otherwise rather than carrying an offline path.
#
# Targets prod by default; override SRC/VENV/BUS/AGENTS via environment to point
# at dev (e.g. SRC=/opt/khonliang-dev/src/khonliang-researcher
# VENV=/opt/khonliang-dev/agents/researcher/.venv BUS=http://localhost:<devport>).
#
# Usage:
#   scripts/deploy.sh [--ref <branch>] [--dry-run] [--no-restart]
#
#   --ref <branch>    git ref to deploy (default: main)
#   --dry-run         print the mutating commands without running them
#   --no-restart      pull + reinstall but leave the running processes untouched
#
set -euo pipefail

# ---- config (override via environment) ------------------------------------
# Space-separated list of bus agent ids to restart after install. Each id is
# matched against the running process by "--id <id>", so it works for both the
# researcher.agent and researcher.librarian_agent modules.
AGENTS="${AGENTS:-researcher-primary librarian-primary}"
SRC="${SRC:-/opt/khonliang/src/khonliang-researcher}"
VENV="${VENV:-/opt/khonliang/agents/researcher/.venv}"
BUS="${BUS:-http://localhost:8788}"
REF="${REF:-main}"
DEPLOY_USER="${DEPLOY_USER:-khonliang}"
# An import that succeeds only when the freshly-installed code is present. Acts
# as a post-install smoke so a half-built install can't silently ship. Imports
# both agent entrypoints' packages plus the worker (embedded distill loop).
VERIFY_IMPORT="${VERIFY_IMPORT:-import researcher.agent, researcher.librarian_agent, researcher.worker; from researcher.pipeline import ResearchPipeline}"

PY="$VENV/bin/python"

# ---- args ------------------------------------------------------------------
DRY_RUN=0
RESTART=1
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) REF="${2:?--ref needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-restart) RESTART=0; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[deploy:error]\033[0m %s\n' "$*" >&2; exit 1; }

# Run a command as the venv-owning user (so new files stay owned by it). Exec
# directly if we already are that user, else via passwordless sudo.
run_as_owner() {
  if [ "$(id -un)" = "$DEPLOY_USER" ]; then
    "$@"
  else
    sudo -n -u "$DEPLOY_USER" "$@"
  fi
}

# Echo a mutating command, then run it unless --dry-run.
do_cmd() {
  log "+ $*"
  [ "$DRY_RUN" = "1" ] && return 0
  "$@"
}

# ---- preflight -------------------------------------------------------------
[ -d "$SRC/.git" ] || die "source clone is not a git repo: $SRC"
[ -x "$PY" ] || die "venv python not found/executable: $PY"
if [ "$(id -un)" != "$DEPLOY_USER" ]; then
  sudo -n true 2>/dev/null || die "need passwordless sudo to act as '$DEPLOY_USER' (or run this script as $DEPLOY_USER)"
fi
# Reachability probe: no -f, so a 404 on the root still counts as "connected".
curl -sS -o /dev/null --max-time 5 "$BUS/" 2>/dev/null || die "bus not reachable at $BUS"

log "agents='$AGENTS' ref=$REF src=$SRC user=$DEPLOY_USER dry_run=$DRY_RUN"

# ---- pull ------------------------------------------------------------------
OLD_SHA="$(run_as_owner git -C "$SRC" rev-parse HEAD)"
do_cmd run_as_owner git -C "$SRC" fetch --quiet origin "$REF"
do_cmd run_as_owner git -C "$SRC" checkout --quiet "$REF"
do_cmd run_as_owner git -C "$SRC" pull --ff-only --quiet origin "$REF"
NEW_SHA="$(run_as_owner git -C "$SRC" rev-parse HEAD)"
if [ "$OLD_SHA" = "$NEW_SHA" ]; then
  log "source already at $NEW_SHA (forcing reinstall + restart anyway)"
else
  log "source $OLD_SHA -> $NEW_SHA"
fi

# ---- reinstall package code (always; cheap, idempotent) --------------------
do_cmd run_as_owner "$PY" -m pip install --force-reinstall --no-deps --no-cache-dir "$SRC"

# ---- sync dependencies only if pyproject.toml changed ----------------------
if [ "$OLD_SHA" != "$NEW_SHA" ] \
   && run_as_owner git -C "$SRC" diff --name-only "$OLD_SHA" "$NEW_SHA" | grep -qx "pyproject.toml"; then
  log "pyproject.toml changed -> syncing dependencies"
  # No --force-reinstall: installs new/changed deps, leaves satisfied ones
  # (so git-sourced libs are not needlessly re-cloned).
  do_cmd run_as_owner "$PY" -m pip install --no-cache-dir "$SRC"
else
  log "no pyproject.toml change -> skipping dependency sync"
fi

# ---- post-install smoke (neutral cwd, so it can't import from a stray dir) --
if [ "$DRY_RUN" != "1" ]; then
  ( cd /tmp && run_as_owner "$PY" -c "$VERIFY_IMPORT" ) \
    && log "install smoke OK" \
    || die "install smoke failed -- new code not importable: $VERIFY_IMPORT"
fi

# ---- restart each agent (bus-managed) --------------------------------------
if [ "$RESTART" = "1" ]; then
  for AGENT in $AGENTS; do
    OLD_PID="$(pgrep -f -- "--id $AGENT" | head -1 || true)"
    do_cmd curl -fsS -X POST "$BUS/v1/install/$AGENT/restart"
    if [ "$DRY_RUN" = "1" ]; then
      log "dry-run: restart of $AGENT skipped"
      continue
    fi
    echo
    NEW_PID=""
    for _ in $(seq 1 30); do
      NEW_PID="$(pgrep -f -- "--id $AGENT" | head -1 || true)"
      [ -n "$NEW_PID" ] && [ "$NEW_PID" != "${OLD_PID:-}" ] && break
      sleep 1
    done
    if [ -n "$NEW_PID" ] && [ "$NEW_PID" != "${OLD_PID:-}" ]; then
      log "restarted $AGENT: pid ${OLD_PID:-none} -> $NEW_PID"
    else
      die "could not confirm a new $AGENT process after restart (old pid=${OLD_PID:-none})"
    fi
  done
else
  log "--no-restart: running processes left untouched (will pick up code on next restart)"
fi

log "deploy complete: [$AGENTS] at $NEW_SHA"
