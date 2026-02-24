#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# create_and_push.sh
#
# Creates a new GitHub repo "dts-scoring-service" under account nitish811
# and pushes the current directory.
#
# Usage:
#   export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
#   bash create_and_push.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GITHUB_USER="nitish811"
REPO_NAME="dts-scoring-service"
DESCRIPTION="Deep Thinking Score (DTS) scoring service — FastAPI + LLaMA-3.1-8B"
PRIVATE=false      # set to true if you want a private repo

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: GITHUB_TOKEN is not set."
  echo "Create a Classic PAT at https://github.com/settings/tokens with 'repo' scope,"
  echo "then run:  export GITHUB_TOKEN=<your-token>"
  exit 1
fi

echo "Creating GitHub repo ${GITHUB_USER}/${REPO_NAME} …"

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/user/repos \
  -d "{
    \"name\": \"${REPO_NAME}\",
    \"description\": \"${DESCRIPTION}\",
    \"private\": ${PRIVATE},
    \"auto_init\": false
  }")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [[ "$HTTP_CODE" -ne 201 ]]; then
  echo "GitHub API error (HTTP $HTTP_CODE):"
  echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
  exit 1
fi

REMOTE_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

# Set remote (or update if it already exists)
if git remote get-url origin &>/dev/null; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

git branch -M main
git push -u origin main

# Swap token out of the remote URL so it isn't stored in plain text
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo ""
echo "✓ Repository live at: https://github.com/${GITHUB_USER}/${REPO_NAME}"
