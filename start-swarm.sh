#!/bin/bash

# Swarm Schrödinger Bridge - Unified Startup Script
# Configured for Port 8080 and www.tree4five.com

set -e

echo "🚀 Launching Swarm Training Node..."
echo "======================================"

# 1. Update Codebase
echo "📥 Pulling latest updates from GitHub..."
git pull origin main

# 2. Install/Update Dependencies
echo "📦 Ensuring dependencies are up to date..."
npm install

# 3. Clean PM2 Environment
echo "🧹 Cleaning up existing processes..."
pm2 delete all || true

# 4. Start Swarm Server
echo "🧠 Starting Swarm Consolidation Server on port 8080..."
# We explicitly set PORT=8080 to override any old .env settings
PORT=8080 pm2 start server/index.js --name swarm-server

# 5. Start Cloudflare Tunnel
echo "☁️ Starting Cloudflare Tunnel (pi-server -> localhost:8080)..."
# Replace 'pi-server' if your tunnel name is different
pm2 start "cloudflared tunnel run --url http://localhost:8080 pi-server" --name cf-tunnel

# 6. Persistence
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
