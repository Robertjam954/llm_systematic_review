#!/usr/bin/env bash
# watch_repo.sh — macOS desktop notifications for remote repo changes
# Usage: bash tools/watch_repo.sh [interval_seconds]
# Requires: git, osascript (macOS built-in)

INTERVAL=${1:-60}
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)

echo "[watch_repo] Watching branch '$BRANCH' every ${INTERVAL}s — Ctrl+C to stop"

while true; do
    git -C "$REPO_DIR" fetch --quiet origin "$BRANCH" 2>/dev/null

    LOCAL=$(git -C "$REPO_DIR" rev-parse HEAD)
    REMOTE=$(git -C "$REPO_DIR" rev-parse "origin/$BRANCH" 2>/dev/null)

    if [[ "$LOCAL" != "$REMOTE" ]]; then
        COMMITS=$(git -C "$REPO_DIR" log --oneline HEAD..origin/"$BRANCH" 2>/dev/null | head -5)
        osascript -e "display notification \"New commits on $BRANCH:\\n$COMMITS\" with title \"llm_systematic_review\" sound name \"Ping\""
        echo "[watch_repo] Remote ahead — new commits detected"
    fi

    sleep "$INTERVAL"
done
