#!/bin/bash
# update-cloudflared.sh
# Update Cloudflared to the latest version on Raspberry Pi

set -e

echo "🔄 Updating Cloudflared on Raspberry Pi"

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

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo -e "${RED}Cloudflared is not installed.${NC}"
    echo "Please run: ./scripts/install-cloudflared.sh"
    exit 1
fi

section "Current Version"
CURRENT_VERSION=$(cloudflared --version | head -n1)
echo "Installed: $CURRENT_VERSION"

# Get latest version info from GitHub
section "Checking for Updates"
echo "Checking latest release on GitHub..."

# Get latest release info
LATEST_INFO=$(curl -s https://api.github.com/repos/cloudflare/cloudflared/releases/latest)
LATEST_VERSION=$(echo "$LATEST_INFO" | grep '"tag_name"' | cut -d'"' -f4)
LATEST_URL=$(echo "$LATEST_INFO" | grep '"browser_download_url"' | grep 'linux-arm64.deb"' | head -1 | cut -d'"' -f4)

if [ -z "$LATEST_URL" ]; then
    # Try ARMv7 version
    LATEST_URL=$(echo "$LATEST_INFO" | grep '"browser_download_url"' | grep 'linux-arm.deb"' | head -1 | cut -d'"' -f4)
fi

if [ -z "$LATEST_VERSION" ] || [ -z "$LATEST_URL" ]; then
    echo -e "${RED}Failed to get latest version information${NC}"
    echo "You can manually download from: https://github.com/cloudflare/cloudflared/releases"
    exit 1
fi

echo "Latest version: $LATEST_VERSION"
echo "Download URL: $LATEST_URL"

# Extract current version number for comparison
CURRENT_VERSION_NUM=$(echo "$CURRENT_VERSION" | grep -oP '\d+\.\d+\.\d+' | head -n1)
LATEST_VERSION_NUM=$(echo "$LATEST_VERSION" | grep -oP '\d+\.\d+\.\d+' | head -n1)

if [ "$CURRENT_VERSION_NUM" = "$LATEST_VERSION_NUM" ]; then
    echo -e "${GREEN}✓ You already have the latest version ($CURRENT_VERSION_NUM)${NC}"
    echo "No update needed."
    exit 0
fi

echo -e "${YELLOW}Update available: $CURRENT_VERSION_NUM → $LATEST_VERSION_NUM${NC}"

section "Backup Current Configuration"

# Backup current config
BACKUP_DIR="/tmp/cloudflared-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Backing up current configuration..."
sudo cp -r /etc/cloudflared "$BACKUP_DIR/" 2>/dev/null || true
sudo cp /home/pi/.cloudflared/*.json "$BACKUP_DIR/" 2>/dev/null || true

echo "Backup created in: $BACKUP_DIR"

section "Stop Cloudflared Service"

echo "Stopping cloudflared service..."
sudo systemctl stop cloudflared 2>/dev/null || true

# Also kill any running cloudflared processes
sudo pkill -f cloudflared 2>/dev/null || true
sleep 2

section "Download and Install Update"

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Download appropriate package
TEMP_FILE="/tmp/cloudflared-latest.deb"
echo "Downloading latest version..."
wget -q "$LATEST_URL" -O "$TEMP_FILE"

if [ ! -f "$TEMP_FILE" ]; then
    echo -e "${RED}Download failed${NC}"
    exit 1
fi

echo "Package downloaded: $(ls -lh "$TEMP_FILE" | awk '{print $5}')"

# Install the new package
echo "Installing update..."
sudo dpkg -i "$TEMP_FILE"

# Fix any dependencies
sudo apt-get install -f -y

# Clean up
rm "$TEMP_FILE"

section "Verify Installation"

NEW_VERSION=$(cloudflared --version | head -n1)
echo "New version: $NEW_VERSION"

if [[ "$NEW_VERSION" == *"$LATEST_VERSION_NUM"* ]]; then
    echo -e "${GREEN}✓ Successfully updated to $LATEST_VERSION_NUM${NC}"
else
    echo -e "${YELLOW}⚠ Version check inconclusive${NC}"
    echo "Installed version reports: $NEW_VERSION"
fi

section "Restore Configuration"

# Restore config files
if [ -d "$BACKUP_DIR" ]; then
    echo "Restoring configuration..."
    sudo cp -r "$BACKUP_DIR/"* /etc/cloudflared/ 2>/dev/null || true
    sudo cp "$BACKUP_DIR/"*.json /home/pi/.cloudflared/ 2>/dev/null || true
    echo "Configuration restored from backup"
fi

section "Start Service"

echo "Starting cloudflared service..."
sudo systemctl daemon-reload
sudo systemctl start cloudflared

sleep 2

# Check service status
SERVICE_STATUS=$(sudo systemctl is-active cloudflared)
if [ "$SERVICE_STATUS" = "active" ]; then
    echo -e "${GREEN}✓ Cloudflared service started successfully${NC}"
else
    echo -e "${RED}✗ Cloudflared service failed to start${NC}"
    echo "Status: $SERVICE_STATUS"
    echo "Checking logs..."
    sudo journalctl -u cloudflared -n 10 --no-pager
fi

section "Test Tunnel"

echo "Testing tunnel connectivity..."
sleep 3

# List tunnels to verify
echo "Active tunnels:"
cloudflared tunnel list 2>/dev/null || echo "Unable to list tunnels"

# Test a simple command
echo -n "Version test: "
cloudflared --version >/dev/null 2>&1 && echo -e "${GREEN}✓ Working${NC}" || echo -e "${RED}✗ Failed${NC}"

section "Update Complete"

echo -e "${GREEN}✅ Cloudflared update completed successfully!${NC}"
echo ""
echo "Summary:"
echo "  • Previous version: $CURRENT_VERSION_NUM"
echo "  • New version:      $LATEST_VERSION_NUM"
echo "  • Service status:   $SERVICE_STATUS"
echo "  • Backup location:  $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "  1. Check service logs: sudo journalctl -u cloudflared -f"
echo "  2. Verify tunnel connectivity: ./scripts/cloudflare-health-check.sh"
echo "  3. Test your application: curl https://your-domain.com/api/health"
echo ""
echo "If you encounter issues:"
echo "  • Check logs: sudo journalctl -u cloudflared -xe"
echo "  • Restore from backup if needed"
echo "  • Re-run setup: ./scripts/setup-cloudflare-tunnel.sh"
echo ""
echo -e "${GREEN}Update process finished at $(date)${NC}"

# Clean up old backups (keep last 3)
echo ""
echo "Cleaning up old backups..."
find /tmp -name "cloudflared-backup-*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
echo "Done."