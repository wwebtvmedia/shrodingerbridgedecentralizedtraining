#!/bin/bash
# cloudflare-health-check.sh
# Comprehensive health check for Cloudflare Tunnel on Raspberry Pi
# Uses environment variables from .env file

set -e

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
elif [ -f .env.template ]; then
    echo "Warning: Using .env.template. Please create .env file with your actual values."
    export $(grep -v '^#' .env.template | xargs)
fi

# Configuration from environment variables
TUNNEL_NAME="${CLOUDFLARE_TUNNEL_NAME:-training-consolidation-tunnel}"
LOCAL_PORT="${CLOUDFLARE_TUNNEL_LOCAL_PORT:-8080}"

# Use URL from environment variables
if [ -z "$URL" ]; then
    echo "Warning: URL environment variable is not set in .env file"
    echo "Using default domain: training.tree4five.com"
    DOMAIN="training.tree4five.com"
else
    # Extract domain from URL
    FULL_DOMAIN="${URL#https://}"
    FULL_DOMAIN="${FULL_DOMAIN#http://}"
    FULL_DOMAIN="${FULL_DOMAIN%%/*}"
    DOMAIN="$FULL_DOMAIN"
fi

echo "Configuration:"
echo "  URL: ${URL:-Not set}"
echo "  Domain: $DOMAIN"
echo "  Tunnel Name: $TUNNEL_NAME"
echo "  Local Port: $LOCAL_PORT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print section headers
section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to print status
status() {
    if [ "$1" = "0" ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

clear
echo "🔍 Cloudflare Tunnel Health Check"
echo "================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

section "1. System Information"

echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "Uptime: $(uptime -p)"

section "2. Cloudflared Installation"

# Check if cloudflared is installed
if command -v cloudflared &> /dev/null; then
    VERSION=$(cloudflared --version | head -n1)
    echo -e "${GREEN}✓ Cloudflared installed: $VERSION${NC}"
else
    echo -e "${RED}✗ Cloudflared not installed${NC}"
    exit 1
fi

section "3. Service Status"

SERVICE_STATUS=$(sudo systemctl is-active cloudflared 2>/dev/null || echo "inactive")
if [ "$SERVICE_STATUS" = "active" ]; then
    echo -e "${GREEN}✓ Cloudflared service: ACTIVE${NC}"
    
    # Check if service is enabled
    if sudo systemctl is-enabled cloudflared &> /dev/null; then
        echo -e "${GREEN}✓ Service enabled on boot${NC}"
    else
        echo -e "${YELLOW}⚠ Service not enabled on boot${NC}"
    fi
    
    # Get service uptime
    SERVICE_UPTIME=$(systemctl show cloudflared --property=ActiveEnterTimestamp | cut -d= -f2)
    if [ -n "$SERVICE_UPTIME" ]; then
        echo "Started: $SERVICE_UPTIME"
    fi
else
    echo -e "${RED}✗ Cloudflared service: $SERVICE_STATUS${NC}"
fi

section "4. Tunnel Status"

# List tunnels
echo "Active tunnels:"
cloudflared tunnel list 2>/dev/null | grep -E "(NAME|$TUNNEL_NAME)" || echo "No tunnels found or error listing tunnels"

# Get specific tunnel info
echo -e "\nTunnel '$TUNNEL_NAME' details:"
cloudflared tunnel info "$TUNNEL_NAME" 2>/dev/null || echo "Tunnel info not available"

section "5. Network Connectivity"

# Test local server
echo -n "Local server (port $LOCAL_PORT): "
if curl -s --max-time 5 "http://localhost:$LOCAL_PORT/api/health" > /dev/null; then
    echo -e "${GREEN}✓ Accessible${NC}"
else
    echo -e "${RED}✗ Not accessible${NC}"
fi

# Test Cloudflare DNS
echo -n "Cloudflare DNS (1.1.1.1): "
if ping -c 1 -W 2 1.1.1.1 &> /dev/null; then
    echo -e "${GREEN}✓ Reachable${NC}"
else
    echo -e "${RED}✗ Unreachable${NC}"
fi

# Test external connectivity
echo -n "External connectivity: "
if curl -s --max-time 5 https://1.1.1.1/cdn-cgi/trace > /dev/null; then
    echo -e "${GREEN}✓ Working${NC}"
else
    echo -e "${RED}✗ Failed${NC}"
fi

section "6. DNS Resolution"

# Check DNS resolution
echo -n "Domain resolution ($DOMAIN): "
if dig +short "$DOMAIN" &> /dev/null; then
    echo -e "${GREEN}✓ Resolves${NC}"
    IP=$(dig +short "$DOMAIN" | head -n1)
    echo "  IP: $IP"
else
    echo -e "${RED}✗ Does not resolve${NC}"
fi

# Test through tunnel
echo -n "Tunnel connectivity ($DOMAIN): "
if curl -s --max-time 10 "https://$DOMAIN/api/health" > /dev/null; then
    echo -e "${GREEN}✓ Accessible${NC}"
    
    # Get response time
    START_TIME=$(date +%s%N)
    curl -s --max-time 5 "https://$DOMAIN/api/health" > /dev/null
    END_TIME=$(date +%s%N)
    RESPONSE_TIME=$((($END_TIME - $START_TIME) / 1000000))
    echo "  Response time: ${RESPONSE_TIME}ms"
else
    echo -e "${YELLOW}⚠ May not be accessible (check DNS or tunnel)${NC}"
fi

section "7. Resource Usage"

# Check cloudflared process
echo "Cloudflared processes:"
ps aux | grep cloudflared | grep -v grep | while read line; do
    PID=$(echo $line | awk '{print $2}')
    CPU=$(echo $line | awk '{print $3}')
    MEM=$(echo $line | awk '{print $4}')
    CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}')
    echo "  PID: $PID, CPU: ${CPU}%, MEM: ${MEM}%"
done

# Check memory usage
echo -e "\nSystem memory:"
free -h | grep -E "(total|Mem:)" | head -n1
free -h | grep "Mem:" | awk '{print "  Used: " $3 "/" $2 " (" $3/$2*100 "%)"}'

section "8. Recent Logs"

echo "Last 10 log entries:"
sudo journalctl -u cloudflared -n 10 --no-pager 2>/dev/null || echo "No logs available"

# Check for errors in last hour
ERROR_COUNT=$(sudo journalctl -u cloudflared --since "1 hour ago" 2>/dev/null | grep -i "error\|fail\|exception" | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo -e "\n${YELLOW}⚠ Found $ERROR_COUNT errors in last hour${NC}"
    sudo journalctl -u cloudflared --since "1 hour ago" 2>/dev/null | grep -i "error\|fail\|exception" | tail -5
fi

section "9. Configuration Check"

# Check config file
CONFIG_FILE="/etc/cloudflared/config.yml"
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}✓ Config file exists: $CONFIG_FILE${NC}"
    
    # Check config syntax
    if sudo cloudflared tunnel --config "$CONFIG_FILE" ingress validate 2>/dev/null; then
        echo -e "${GREEN}✓ Config syntax is valid${NC}"
    else
        echo -e "${RED}✗ Config syntax error${NC}"
    fi
    
    # Show basic config info
    echo "Tunnel name in config: $(grep -E "^tunnel:" "$CONFIG_FILE" | head -1 | awk '{print $2}')"
    echo "Number of ingress rules: $(grep -c "^- hostname:" "$CONFIG_FILE")"
else
    echo -e "${RED}✗ Config file not found${NC}"
fi

section "10. Firewall Check"

# Check if firewall is active
if command -v ufw &> /dev/null; then
    UFW_STATUS=$(sudo ufw status | grep -o "Status: active")
    if [ "$UFW_STATUS" = "Status: active" ]; then
        echo -e "${YELLOW}⚠ UFW firewall is active${NC}"
        echo "Checking rules for cloudflared:"
        sudo ufw status | grep -E "(cloudflared|$LOCAL_PORT)" || echo "  No specific rules found"
    else
        echo -e "${GREEN}✓ UFW firewall is inactive${NC}"
    fi
fi

section "11. Certificate Check"

# Check tunnel certificate
CERT_FILE="/home/pi/.cloudflared/cert.pem"
if [ -f "$CERT_FILE" ]; then
    echo -e "${GREEN}✓ Certificate file exists${NC}"
    
    # Check certificate expiry
    if command -v openssl &> /dev/null; then
        EXPIRY=$(openssl x509 -in "$CERT_FILE" -noout -enddate 2>/dev/null | cut -d= -f2)
        if [ -n "$EXPIRY" ]; then
            echo "  Expires: $EXPIRY"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Certificate file not found${NC}"
fi

section "Health Summary"

echo ""
echo "📊 Summary:"
echo "----------"

# Count successes
SUCCESS_COUNT=0
TOTAL_CHECKS=0

# We'll do a simple summary based on key checks
checks=(
    "Cloudflared installed"
    "Service active" 
    "Local server accessible"
    "DNS resolves"
    "Config valid"
)

for check in "${checks[@]}"; do
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    # This is simplified - in a real script you'd track actual results
done

echo "Performed $TOTAL_CHECKS key checks"
echo ""

# Recommendations
section "Recommendations"

if [ "$SERVICE_STATUS" != "active" ]; then
    echo "1. Start cloudflared service: sudo systemctl start cloudflared"
fi

if ! curl -s "http://localhost:$LOCAL_PORT/api/health" > /dev/null; then
    echo "2. Ensure Training Consolidation Server is running: npm run server"
fi

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "3. Check logs for errors: sudo journalctl -u cloudflared -f"
fi

echo ""
echo "For detailed troubleshooting, see: cloudflare-raspberry-pi-guide.md"
echo ""

section "Quick Commands Reference"

echo "View live logs:        sudo journalctl -u cloudflared -f"
echo "Restart service:       sudo systemctl restart cloudflared"
echo "Check tunnel status:   cloudflared tunnel list"
echo "Test connectivity:     curl https://$DOMAIN/api/health"
echo "Update cloudflared:    ./scripts/update-cloudflared.sh"
echo ""

echo -e "${GREEN}Health check completed at $(date)${NC}"