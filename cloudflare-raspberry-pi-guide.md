# Complete Cloudflare Integration Guide for Raspberry Pi

This guide provides comprehensive instructions for using Cloudflare services with your Raspberry Pi running the Training Consolidation Server.

## Overview

Cloudflare offers several services that are particularly useful for Raspberry Pi deployments:

1. **Cloudflare Tunnel** - Secure outbound connection without opening ports
2. **Cloudflare DNS** - Domain management and DNS records
3. **Cloudflare Workers** - Serverless functions for edge computing
4. **Cloudflare Access** - Zero-trust security for your applications
5. **Cloudflare DDoS Protection** - Protection against attacks

## Prerequisites

- Raspberry Pi with Raspberry Pi OS (Bullseye or newer)
- Node.js 18+ installed (already done in your setup)
- Cloudflare account with a domain
- Training Consolidation Server running on port 8080

## Environment Configuration

Before starting, configure your environment variables:

1. **Copy the template file**:

   ```bash
   cp .env.template .env
   ```

2. **Edit the `.env` file** with your values:

   ```bash
   nano .env
   ```

3. **Key variables to set**:

   ```bash
   # Your public URL (required)
   URL=https://training.yourdomain.com

   # Cloudflare Tunnel configuration (optional, defaults shown)
   CLOUDFLARE_TUNNEL_NAME=training-consolidation-tunnel
   CLOUDFLARE_TUNNEL_SUBDOMAIN=training
   CLOUDFLARE_TUNNEL_LOCAL_PORT=8080

   # Server configuration
   SERVER_PORT=8080
   NODE_ENV=production
   ```

4. **Important**: The `.env` file is excluded from git (see `.gitignore`). Never commit sensitive information.

The `URL` variable is the primary configuration. Scripts will automatically parse it to extract:

- Domain (e.g., `yourdomain.com`)
- Subdomain (e.g., `training`)
- Full domain (e.g., `training.yourdomain.com`)

## 1. Cloudflare Tunnel Setup (Recommended)

### Step 1: Install Cloudflared on Raspberry Pi

```bash
#!/bin/bash
# install-cloudflared.sh

echo "Installing Cloudflared on Raspberry Pi..."

# Download the ARM64 version of cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb

# Install the package
sudo dpkg -i cloudflared-linux-arm64.deb

# Verify installation
cloudflared --version

# Clean up
rm cloudflared-linux-arm64.deb

echo "Cloudflared installation complete!"
```

### Step 2: Authenticate with Cloudflare

```bash
# This will open a browser URL for authentication
cloudflared tunnel login
```

If you're running headless (no browser), use:

```bash
cloudflared tunnel login --url-only
# Copy the URL and open it on another device with browser access
```

### Step 3: Create a Tunnel

```bash
# Create a new tunnel
cloudflared tunnel create training-consolidation-tunnel

# Note the tunnel ID displayed (e.g., 6ff42ae2-765d-4adf-8112-31c55c1551ef)
```

### Step 4: Configure the Tunnel

Create configuration file:

```bash
sudo mkdir -p /etc/cloudflared
sudo nano /etc/cloudflared/config.yml
```

Add the following configuration (replace with your actual tunnel ID):

```yaml
# /etc/cloudflared/config.yml
tunnel: training-consolidation-tunnel
credentials-file: /home/pi/.cloudflared/<tunnel-id>.json

# Logging settings
logfile: /var/log/cloudflared.log
loglevel: info

# Ingress rules
ingress:
  # Route web interface
  - hostname: training.tree4five.com
    service: http://localhost:8080

  # Route API endpoints
  - hostname: api.training.tree4five.com
    service: http://localhost:8080/api

  # Route WebSocket
  - hostname: ws.training.tree4five.com
    service: http://localhost:8080

  # Catch-all rule (404)
  - service: http_status:404
```

### Step 5: Run as System Service

```bash
# Install cloudflared as a system service
sudo cloudflared service install

# Start the service
sudo systemctl start cloudflared

# Enable auto-start on boot
sudo systemctl enable cloudflared

# Check service status
sudo systemctl status cloudflared

# View logs
sudo journalctl -u cloudflared -f
```

### Step 6: Configure DNS Records

In your Cloudflare dashboard:

1. Go to DNS → Records
2. Create CNAME records:
   - Name: `training` → Target: `<tunnel-id>.cfargotunnel.com`
   - Name: `api.training` → Target: `<tunnel-id>.cfargotunnel.com`
   - Name: `ws.training` → Target: `<tunnel-id>.cfargotunnel.com`

### Step 7: Verify Tunnel Status

```bash
# List all tunnels
cloudflared tunnel list

# Get tunnel info
cloudflared tunnel info training-consolidation-tunnel

# Test connectivity
curl https://training.tree4five.com/api/health
```

## 2. Complete Setup Script

Here's a complete setup script that automates everything:

```bash
#!/bin/bash
# setup-cloudflare-tunnel.sh

set -e

echo "🚀 Setting up Cloudflare Tunnel for Training Consolidation Server"

# Configuration
DOMAIN="tree4five.com"
SUBDOMAIN="training"
TUNNEL_NAME="training-consolidation-tunnel"
LOCAL_PORT="8080"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Installing Cloudflared...${NC}"

# Check if cloudflared is already installed
if command -v cloudflared &> /dev/null; then
    echo "Cloudflared is already installed"
else
    echo "Downloading Cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
    sudo dpkg -i cloudflared-linux-arm64.deb
    rm cloudflared-linux-arm64.deb
fi

echo -e "${YELLOW}Step 2: Authenticating with Cloudflare...${NC}"
echo "Please authenticate with Cloudflare..."
cloudflared tunnel login

echo -e "${YELLOW}Step 3: Creating tunnel '$TUNNEL_NAME'...${NC}"
TUNNEL_OUTPUT=$(cloudflared tunnel create "$TUNNEL_NAME")
TUNNEL_ID=$(echo "$TUNNEL_OUTPUT" | grep -oP '(?<=Tunnel ID: )\S+')

if [ -z "$TUNNEL_ID" ]; then
    echo -e "${RED}Failed to extract tunnel ID${NC}"
    exit 1
fi

echo "Tunnel created with ID: $TUNNEL_ID"

echo -e "${YELLOW}Step 4: Creating configuration...${NC}"

# Create config directory
sudo mkdir -p /etc/cloudflared

# Create config file
sudo tee /etc/cloudflared/config.yml > /dev/null <<EOF
tunnel: $TUNNEL_NAME
credentials-file: /home/pi/.cloudflared/$TUNNEL_ID.json

logfile: /var/log/cloudflared.log
loglevel: info

ingress:
  # Main web interface
  - hostname: $SUBDOMAIN.$DOMAIN
    service: http://localhost:$LOCAL_PORT

  # API endpoints
  - hostname: api.$SUBDOMAIN.$DOMAIN
    service: http://localhost:$LOCAL_PORT/api

  # WebSocket endpoint
  - hostname: ws.$SUBDOMAIN.$DOMAIN
    service: http://localhost:$LOCAL_PORT

  # Health check endpoint
  - hostname: health.$SUBDOMAIN.$DOMAIN
    service: http://localhost:$LOCAL_PORT/api/health

  # Catch-all 404
  - service: http_status:404
EOF

echo -e "${YELLOW}Step 5: Setting up as system service...${NC}"

# Install service
sudo cloudflared service install

# Start and enable service
sudo systemctl daemon-reload
sudo systemctl start cloudflared
sudo systemctl enable cloudflared

echo -e "${YELLOW}Step 6: Creating DNS records...${NC}"
echo "Please create the following DNS records in your Cloudflare dashboard:"
echo "1. CNAME: $SUBDOMAIN.$DOMAIN → $TUNNEL_ID.cfargotunnel.com"
echo "2. CNAME: api.$SUBDOMAIN.$DOMAIN → $TUNNEL_ID.cfargotunnel.com"
echo "3. CNAME: ws.$SUBDOMAIN.$DOMAIN → $TUNNEL_ID.cfargotunnel.com"
echo "4. CNAME: health.$SUBDOMAIN.$DOMAIN → $TUNNEL_ID.cfargotunnel.com"

echo -e "${YELLOW}Step 7: Testing the tunnel...${NC}"

# Wait for service to start
sleep 5

# Check service status
if sudo systemctl is-active --quiet cloudflared; then
    echo -e "${GREEN}✓ Cloudflared service is running${NC}"
else
    echo -e "${RED}✗ Cloudflared service failed to start${NC}"
    sudo journalctl -u cloudflared -n 20 --no-pager
fi

# Show tunnel info
echo -e "\n${YELLOW}Tunnel Information:${NC}"
cloudflared tunnel info "$TUNNEL_NAME"

echo -e "\n${GREEN}✅ Cloudflare Tunnel setup complete!${NC}"
echo -e "Your Training Consolidation Server will be accessible at:"
echo -e "  • Web Interface: https://$SUBDOMAIN.$DOMAIN"
echo -e "  • API: https://api.$SUBDOMAIN.$DOMAIN"
echo -e "  • WebSocket: wss://ws.$SUBDOMAIN.$DOMAIN"
echo -e "\nTo monitor logs: sudo journalctl -u cloudflared -f"
```

## 3. Monitoring and Maintenance Scripts

### Health Check Script

```bash
#!/bin/bash
# cloudflare-health-check.sh

TUNNEL_NAME="training-consolidation-tunnel"
DOMAIN="training.tree4five.com"

echo "🔍 Cloudflare Tunnel Health Check"
echo "================================="

# Check if service is running
if sudo systemctl is-active --quiet cloudflared; then
    echo "✓ Cloudflared service: RUNNING"
else
    echo "✗ Cloudflared service: STOPPED"
    exit 1
fi

# Check tunnel status
echo -e "\n📊 Tunnel Status:"
cloudflared tunnel list | grep "$TUNNEL_NAME"

# Test connectivity
echo -e "\n🌐 Testing Connectivity:"
if curl -s --max-time 10 "https://$DOMAIN/api/health" > /dev/null; then
    echo "✓ Web Interface: ACCESSIBLE"
else
    echo "✗ Web Interface: UNREACHABLE"
fi

# Check logs for errors
echo -e "\n📋 Recent Logs (last 5 lines):"
sudo journalctl -u cloudflared -n 5 --no-pager

# Check resource usage
echo -e "\n💾 Resource Usage:"
ps aux | grep cloudflared | grep -v grep
```

### Update Script

```bash
#!/bin/bash
# update-cloudflared.sh

echo "🔄 Updating Cloudflared..."

# Stop service
sudo systemctl stop cloudflared

# Download latest version
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb

# Install update
sudo dpkg -i cloudflared-linux-arm64.deb

# Clean up
rm cloudflared-linux-arm64.deb

# Start service
sudo systemctl start cloudflared

echo "✅ Cloudflared updated successfully!"
echo "Current version: $(cloudflared --version)"
```

## 4. Advanced Configuration

### Multiple Services Configuration

If you're running multiple services on your Raspberry Pi:

```yaml
# /etc/cloudflared/config.yml
tunnel: raspberry-pi-tunnel
credentials-file: /home/pi/.cloudflared/tunnel-id.json

ingress:
  # Training Consolidation Server
  - hostname: training.tree4five.com
    service: http://localhost:8080

  # Home Assistant (if running)
  - hostname: home.tree4five.com
    service: http://localhost:8123

  # Pi-hole DNS (if running)
  - hostname: pihole.tree4five.com
    service: http://localhost:80

  # SSH over HTTPS (Cloudflare Access)
  - hostname: ssh.tree4five.com
    service: ssh://localhost:22

  # Catch-all
  - service: http_status:404
```

### Load Balancing Configuration

For high availability with multiple Raspberry Pis:

```yaml
ingress:
  - hostname: training.tree4five.com
    service: http://localhost:8080
    originRequest:
      noTLSVerify: true

  # Load balancer pool
  - hostname: lb.training.tree4five.com
    service: lb-pool:training-servers

# Load balancer configuration
loadBalancer:
  pools:
    - name: training-servers
      origins:
        - address: http://192.168.1.100:8080
          weight: 1
        - address: http://192.168.1.101:8080
          weight: 1
        - address: http://192.168.1.102:8080
          weight: 1
```

## 5. Security Enhancements

### Cloudflare Access Integration

Add zero-trust security to your application:

1. In Cloudflare Dashboard, go to Access → Applications
2. Create a new application for `training.tree4five.com`
3. Configure policies (e.g., require email domain, 2FA)
4. Update your tunnel configuration:

```yaml
ingress:
  - hostname: training.tree4five.com
    service: http://localhost:8080
    originRequest:
      access:
        teamName: your-team-name
        audTag: your-aud-tag
```

### Rate Limiting

Protect your Raspberry Pi from abuse:

1. In Cloudflare Dashboard, go to Security → WAF → Rate limiting rules
2. Create rules like:
   - Block more than 100 requests per minute from a single IP
   - Allowlist your own IP addresses
   - Set challenge for suspicious traffic

## 6. Performance Optimization

### Caching Configuration

Add caching for static assets:

```yaml
ingress:
  - hostname: training.tree4five.com
    service: http://localhost:8080
    originRequest:
      # Cache static assets
      cacheSettings:
        cacheEverything: false
        cacheKey:
          ignoreQueryStrings: true
          headers:
            include:
              - Origin
              - Accept
```

### Compression

Enable compression for faster loading:

```bash
# In your server configuration (server/index.js), add:
app.use(compression());
```

## 7. Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Tunnel won't start

```bash
# Check logs
sudo journalctl -u cloudflared -xe

# Common fix: Reinstall credentials
cloudflared tunnel cleanup
cloudflared tunnel login
```

#### Issue 2: DNS not resolving

```bash
# Check DNS propagation
dig training.tree4five.com

# Verify Cloudflare proxy is enabled (orange cloud)
# In Cloudflare DNS settings, ensure proxy status is "Proxied"
```

#### Issue 3: High memory usage

```bash
# Monitor memory
htop

# Restart service
sudo systemctl restart cloudflared

# Check for memory leaks
sudo journalctl -u cloudflared | grep -i "memory\|oom"
```

#### Issue 4: Connection timeouts

```bash
# Test local connectivity
curl http://localhost:8080/api/health

# Check firewall
sudo ufw status

# Test tunnel connectivity
cloudflared tunnel test training-consolidation-tunnel
```

### Diagnostic Script

```bash
#!/bin/bash
# cloudflare-diagnostic.sh

echo "🔧 Cloudflare Tunnel Diagnostic"
echo "==============================="

echo "1. System Information:"
uname -a
cat /etc/os-release | grep PRETTY_NAME

echo -e "\n2. Cloudflared Version:"
cloudflared --version

echo -e "\n3. Service Status:"
sudo systemctl status cloudflared --no-pager

echo -e "\n4. Active Tunnels:"
cloudflared tunnel list

echo -e "\n5. Network Connectivity:"
ping -c 3 1.1.1.1
curl -s --max-time 5 https://1.1.1.1/cdn-cgi/trace

echo -e "\n6. Local Server Check:"
curl -s http://localhost:8080/api/health || echo "Local server not responding"

echo -e "\n7. Recent Errors:"
sudo journalctl -u cloudflared --since "1 hour ago" | grep -i "error\|fail\|warn"

echo -e "\n8. Resource Usage:"
ps aux | grep cloudflared | grep -v grep
free -h
```

## 8. Integration with Your Training System

### WebSocket Configuration

For real-time training updates, ensure WebSocket support:

```javascript
// In your client code (src/consolidation-client.js)
const wsUrl =
  window.location.protocol === "https:"
    ? `wss://${window.location.host}`
    : `ws://${window.location.host}`;

const socket = new WebSocket(wsUrl);
```

### Environment Configuration

Create environment-specific configuration:

```bash
# .env.cloudflare
CLOUDFLARE_TUNNEL_ENABLED=true
PUBLIC_URL=https://training.tree4five.com
WS_URL=wss://ws.training.tree4five.com
API_URL=https://api.training.tree4five.com
```

Update your server to use these variables:

```javascript
// server/index.js
const config = {
  port: process.env.PORT || 8080,
  publicUrl:
    process.env.PUBLIC_URL || `http://localhost:${process.env.PORT || 8080}`,
  wsUrl: process.env.WS_URL || `ws://localhost:${process.env.PORT || 8080}`,
  cloudflareTunnel: process.env.CLOUDFLARE_TUNNEL_ENABLED === "true",
};
```

## 9. Alternative Cloudflare Services for Raspberry Pi

### Cloud
