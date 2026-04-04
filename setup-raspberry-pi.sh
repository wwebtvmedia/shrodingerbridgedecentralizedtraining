#!/bin/bash

# Swarm Schrödinger Bridge - Raspberry Pi Full Setup Script
# Target Domain: www.tree4five.com

set -e

echo "🚀 Starting Full Raspberry Pi Setup..."
echo "======================================"

# 1. Update System
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl git build-essential

# 2. Install Node.js (v20 LTS)
echo "🟢 Installing Node.js v20..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 3. Install cloudflared (Cloudflare Tunnel)
echo "☁️ Installing cloudflared..."
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared-ascii.repo.s3.amazonaws.com/ cloudflared main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install -y cloudflared

# 4. Install Project Dependencies
echo "📂 Installing project dependencies..."
npm install

# 5. Setup Environment Variables
echo "📝 Configuring .env for www.tree4five.com..."
if [ ! -f .env ]; then
    cp .env.template .env
    # Update the URL in .env (handling different possible key names)
    sed -i 's|TUNNEL_URL=.*|TUNNEL_URL=https://www.tree4five.com|g' .env
    sed -i 's|PORT=.*|PORT=3001|g' .env
    echo "✅ Created .env from template"
else
    echo "ℹ️ .env already exists, skipping creation"
fi

# 6. Install PM2 for Process Management
echo "🔄 Installing PM2..."
sudo npm install -g pm2

# 7. Final Instructions
echo "======================================"
echo "✅ Base installation complete!"
echo ""
echo "👉 NEXT STEPS (Manual Action Required):"
echo "1. Authenticate Cloudflare:"
echo "   cloudflared tunnel login"
echo ""
echo "2. Create your tunnel (e.g., named 'pi-server'):"
echo "   cloudflared tunnel create pi-server"
echo ""
echo "3. Route your domain:"
echo "   cloudflared tunnel route dns pi-server www.tree4five.com"
echo ""
echo "4. Start the server with PM2:"
echo "   pm2 start server/index.js --name swarm-server"
echo ""
echo "5. Run the tunnel with PM2:"
echo "   pm2 start \"cloudflared tunnel run pi-server\" --name cf-tunnel"
echo ""
echo "6. Save PM2 configuration to start on boot:"
echo "   pm2 save && pm2 startup"
echo "======================================"
