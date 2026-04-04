#!/bin/bash
# setup-cloudflare-tunnel.sh
# Complete Cloudflare Tunnel setup for Training Consolidation Server
# Uses environment variables from .env file

set -e

echo "🚀 Setting up Cloudflare Tunnel for Training Consolidation Server"

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
elif [ -f .env.template ]; then
    echo "Warning: Using .env.template. Please create .env file with your actual values."
    echo "Copy .env.template to .env and edit with your values."
    export $(grep -v '^#' .env.template | xargs)
else
    echo "Error: No .env or .env.template file found."
    echo "Please create .env file with your configuration."
    exit 1
fi

# Parse URL to extract domain and subdomain
if [ -z "$URL" ]; then
    echo "Error: URL environment variable is not set in .env file"
    echo "Please set URL=https://yourdomain.com in .env"
    exit 1
fi

# Extract domain from URL (remove https:// and path)
FULL_DOMAIN="${URL#https://}"
FULL_DOMAIN="${FULL_DOMAIN#http://}"
FULL_DOMAIN="${FULL_DOMAIN%%/*}"

# Check if it's a subdomain or root domain
if [[ "$FULL_DOMAIN" == *.*.* ]]; then
    # Has subdomain (e.g., training.domain.com)
    SUBDOMAIN="${FULL_DOMAIN%%.*}"
    DOMAIN="${FULL_DOMAIN#*.}"
else
    # Root domain (e.g., domain.com)
    SUBDOMAIN=""
    DOMAIN="$FULL_DOMAIN"
fi

# Configuration from environment variables
TUNNEL_NAME="${CLOUDFLARE_TUNNEL_NAME:-training-consolidation-tunnel}"
LOCAL_PORT="${CLOUDFLARE_TUNNEL_LOCAL_PORT:-8080}"
USER="${USER:-pi}"

# If no subdomain in URL, use the configured one
if [ -z "$SUBDOMAIN" ]; then
    SUBDOMAIN="${CLOUDFLARE_TUNNEL_SUBDOMAIN:-training}"
    FULL_DOMAIN="$SUBDOMAIN.$DOMAIN"
fi

echo "Configuration loaded:"
echo "  URL: $URL"
echo "  Domain: $DOMAIN"
echo "  Subdomain: $SUBDOMAIN"
echo "  Full Domain: $FULL_DOMAIN"
echo "  Tunnel Name: $TUNNEL_NAME"
echo "  Local Port: $LOCAL_PORT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1${NC}"
        exit 1
    fi
}

section "Prerequisites Check"

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo -e "${RED}Cloudflared is not installed.${NC}"
    echo "Please run: ./scripts/install-cloudflared.sh"
    exit 1
fi

# Check if server is running locally
if curl -s --max-time 2 "http://localhost:$LOCAL_PORT/api/health" > /dev/null; then
    echo -e "${GREEN}✓ Local server is running on port $LOCAL_PORT${NC}"
else
    echo -e "${YELLOW}⚠ Local server not detected on port $LOCAL_PORT${NC}"
    echo "Make sure your Training Consolidation Server is running."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

section "Step 1: Authenticate with Cloudflare"

echo "You need to authenticate with your Cloudflare account."
echo "This will open a browser URL (or show it if running headless)."
echo ""
read -p "Press Enter to continue..."

# Try to authenticate
cloudflared tunnel login
check_success "Cloudflare authentication"

section "Step 2: Create Tunnel"

echo "Creating tunnel: $TUNNEL_NAME"
TUNNEL_OUTPUT=$(cloudflared tunnel create "$TUNNEL_NAME")

# Extract tunnel ID
TUNNEL_ID=$(echo "$TUNNEL_OUTPUT" | grep -oP '(?<=Tunnel ID: )\S+' | head -n1)

if [ -z "$TUNNEL_ID" ]; then
    # Try alternative extraction method
    TUNNEL_ID=$(echo "$TUNNEL_OUTPUT" | grep -oP '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}' | head -n1)
fi

if [ -z "$TUNNEL_ID" ]; then
    echo -e "${RED}Failed to extract tunnel ID from output:${NC}"
    echo "$TUNNEL_OUTPUT"
    exit 1
fi

echo -e "${GREEN}✓ Tunnel created with ID: $TUNNEL_ID${NC}"

section "Step 3: Create Configuration"

# Create config directory
sudo mkdir -p /etc/cloudflared
check_success "Created /etc/cloudflared directory"

# Create config file
CONFIG_FILE="/etc/cloudflared/config.yml"
sudo tee "$CONFIG_FILE" > /dev/null <<EOF
# Cloudflare Tunnel Configuration
# Generated: $(date)
# For: Training Consolidation Server

tunnel: $TUNNEL_NAME
credentials-file: /home/$USER/.cloudflared/$TUNNEL_ID.json

# Logging
logfile: /var/log/cloudflared.log
loglevel: info

# Metrics (optional)
metrics: 0.0.0.0:20000
tag: project=training-consolidation

# Ingress rules
ingress:
  # Main web interface
  - hostname: $SUBDOMAIN.$DOMAIN
    service: http://localhost:$LOCAL_PORT
    originRequest:
      connectTimeout: 30s
      noTLSVerify: false
      httpHostHeader: $SUBDOMAIN.$DOMAIN

  # API endpoints
  - hostname: api.$SUBDOMAIN.$DOMAIN
    service: http://localhost:$LOCAL_PORT/api
    originRequest:
      connectTimeout: 30s

  # WebSocket endpoint
  - hostname: ws.$SUBDOMAIN.$DOMAIN
    service: http://localhost:$LOCAL_PORT
    originRequest:
      connectTimeout: 30s

  # Health check endpoint
  - hostname: health.$SUBDOMAIN.$DOMAIN
    service: http://localhost:$LOCAL_PORT/api/health
    originRequest:
      connectTimeout: 10s

  # Catch-all rule (must be last)
  - service: http_status:404
EOF

check_success "Created configuration file: $CONFIG_FILE"

# Set proper permissions
sudo chmod 644 "$CONFIG_FILE"
sudo chown root:root "$CONFIG_FILE"

section "Step 4: Install as System Service"

echo "Installing cloudflared as a system service..."
sudo cloudflared service install
check_success "Installed cloudflared service"

# Reload systemd
sudo systemctl daemon-reload
check_success "Reloaded systemd daemon"

# Start the service
sudo systemctl start cloudflared
check_success "Started cloudflared service"

# Enable auto-start on boot
sudo systemctl enable cloudflared
check_success "Enabled cloudflared auto-start"

section "Step 5: Verify Service Status"

SERVICE_STATUS=$(sudo systemctl is-active cloudflared)
if [ "$SERVICE_STATUS" = "active" ]; then
    echo -e "${GREEN}✓ Cloudflared service is running${NC}"
else
    echo -e "${RED}✗ Cloudflared service is not running: $SERVICE_STATUS${NC}"
    echo "Checking logs..."
    sudo journalctl -u cloudflared -n 10 --no-pager
fi

# Show tunnel info
echo -e "\n${YELLOW}Tunnel Information:${NC}"
cloudflared tunnel info "$TUNNEL_NAME" 2>/dev/null || echo "Tunnel info not available yet"

section "Step 6: DNS Configuration Instructions"

echo -e "${YELLOW}IMPORTANT:${NC} You need to create DNS records in your Cloudflare dashboard."
echo ""
echo "Go to: https://dash.cloudflare.com → Your Domain → DNS → Records"
echo ""
echo "Create the following CNAME records (all pointing to your tunnel):"
echo ""
echo "Type    Name                      Target"
echo "----    --------------------      ---------------------------------"
echo "CNAME   $SUBDOMAIN.$DOMAIN        $TUNNEL_ID.cfargotunnel.com"
echo "CNAME   api.$SUBDOMAIN.$DOMAIN    $TUNNEL_ID.cfargotunnel.com"
echo "CNAME   ws.$SUBDOMAIN.$DOMAIN     $TUNNEL_ID.cfargotunnel.com"
echo "CNAME   health.$SUBDOMAIN.$DOMAIN $TUNNEL_ID.cfargotunnel.com"
echo ""
echo "Make sure each record has:"
echo "  • Proxy status: █ Proxied (orange cloud)"
echo "  • TTL: Auto"
echo ""

section "Step 7: Testing"

echo "Waiting for DNS propagation (30 seconds)..."
sleep 30

echo -e "\n${YELLOW}Testing connectivity:${NC}"

# Test local connection
echo -n "Local server: "
if curl -s --max-time 5 "http://localhost:$LOCAL_PORT/api/health" > /dev/null; then
    echo -e "${GREEN}✓ Accessible${NC}"
else
    echo -e "${RED}✗ Not accessible${NC}"
fi

# Test through tunnel (if DNS is set up)
echo -n "Through tunnel ($SUBDOMAIN.$DOMAIN): "
if curl -s --max-time 10 "https://$SUBDOMAIN.$DOMAIN/api/health" > /dev/null; then
    echo -e "${GREEN}✓ Accessible${NC}"
else
    echo -e "${YELLOW}⚠ May not be accessible yet (DNS propagation or not configured)${NC}"
fi

section "Setup Complete!"

echo -e "${GREEN}✅ Cloudflare Tunnel setup completed successfully!${NC}"
echo ""
echo "Your Training Consolidation Server is now accessible via:"
echo ""
echo -e "  ${BLUE}• Web Interface:${NC}  https://$SUBDOMAIN.$DOMAIN"
echo -e "  ${BLUE}• API:${NC}            https://api.$SUBDOMAIN.$DOMAIN"
echo -e "  ${BLUE}• WebSocket:${NC}      wss://ws.$SUBDOMAIN.$DOMAIN"
echo -e "  ${BLUE}• Health Check:${NC}   https://health.$SUBDOMAIN.$DOMAIN/api/health"
echo ""
echo "Useful commands:"
echo "  • View logs:              sudo journalctl -u cloudflared -f"
echo "  • Check status:           sudo systemctl status cloudflared"
echo "  • List tunnels:           cloudflared tunnel list"
echo "  • Tunnel info:            cloudflared tunnel info $TUNNEL_NAME"
echo "  • Restart service:        sudo systemctl restart cloudflared"
echo "  • Update cloudflared:     ./scripts/update-cloudflared.sh"
echo ""
echo "For troubleshooting, see: cloudflare-raspberry-pi-guide.md#7-troubleshooting-guide"
echo ""
echo -e "${GREEN}🎉 Your decentralized training system is now publicly accessible!${NC}"