#!/bin/bash

# Swarm Schrödinger Bridge - Unified Startup Script
# Configured for Port 8080 and www.tree4five.com

set -e

# Error handling to keep window open
trap 'echo "❌ An error occurred. Press Enter to close..."; read' ERR

echo "🚀 Launching Swarm Training Node..."
echo "======================================"

# 1. Update Codebase
echo "📥 Pulling latest updates from GitHub..."
# Do NOT blindly `git reset --hard` + `git clean -fd`: that silently destroys
# uncommitted changes AND untracked runtime data (.env, data/, models/,
# checkpoints/, logs). Instead, stash local changes so they can be recovered,
# and only force-discard when FORCE_RESET=1 is explicitly set.
if [ "${FORCE_RESET:-0}" = "1" ]; then
  echo "⚠️  FORCE_RESET=1 — discarding local changes (untracked data preserved)."
  git reset --hard
  # -d removes untracked dirs but -e keeps runtime data out of harm's way.
  git clean -fd -e .env -e data -e models -e checkpoints -e "*.log"
else
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "📦 Stashing local changes (recover with 'git stash pop')..."
    git stash push -u -m "auto-stash before swarm startup" || true
  fi
fi
git pull origin main

# 2. Install/Update Dependencies
echo "📦 Ensuring dependencies are up to date..."
npm install

# 3. Load environment / verify auth secret
# The server authenticates protected endpoints against SECRET_TOKEN. Load .env
# if present so pm2 inherits it, and warn if no token is configured (the server
# will then run with a random ephemeral token printed to its log).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
if [ -z "${SECRET_TOKEN:-}" ]; then
  echo "⚠️  SECRET_TOKEN is not set. Set it in .env (openssl rand -hex 32) for a"
  echo "    stable shared secret; otherwise the server uses a random per-run token."
fi

# 4. Clean PM2 Environment
echo "🧹 Cleaning up existing processes..."
pm2 delete all || true

# 5. Start Swarm Server
echo "🧠 Starting Swarm Consolidation Server on port 8080..."
# We explicitly set PORT=8080 to override any old .env settings.
# --update-env ensures pm2 picks up SECRET_TOKEN/ALLOWED_ORIGINS from above.
PORT=8080 pm2 start server/index.js --name swarm-server --update-env

# 6. Start Cloudflare Tunnel
echo "☁️ Starting Cloudflare Tunnel (pi-server -> localhost:8080)..."
# Replace 'pi-server' if your tunnel name is different
pm2 start "cloudflared tunnel run --url http://localhost:8080 pi-server" --name cf-tunnel

# 7. Persistence
echo "💾 Saving PM2 process list for automatic reboot..."
pm2 save

echo "======================================"
echo "✅ Startup complete!"
echo ""
echo "📊 Current Status:"
pm2 status

echo ""
echo "🔍 Verification Commands:"
echo "1. Check server logs: pm2 logs swarm-server"
echo "2. Check tunnel logs: pm2 logs cf-tunnel"
echo "3. Verify port 8080:  ss -nlt | grep 8080"
echo ""
echo "🌐 Access your site at: https://www.tree4five.com/enhanced-index.html"
echo "======================================"
