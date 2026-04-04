#!/bin/bash
# install-cloudflared.sh
# Install Cloudflared on Raspberry Pi (ARM64)

set -e

echo "🚀 Installing Cloudflared on Raspberry Pi..."

# Check if already installed
if command -v cloudflared &> /dev/null; then
    CURRENT_VERSION=$(cloudflared --version | head -n1)
    echo "Cloudflared is already installed: $CURRENT_VERSION"
    echo "To update, run: ./scripts/update-cloudflared.sh"
    exit 0
fi

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Download appropriate version
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    echo "Downloading ARM64 version..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb -O cloudflared-latest.deb
elif [[ "$ARCH" == "armv7l" || "$ARCH" == "armhf" ]]; then
    echo "Downloading ARMv7 version..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm.deb -O cloudflared-latest.deb
else
    echo "Unsupported architecture: $ARCH"
    echo "Please download manually from: https://github.com/cloudflare/cloudflared/releases"
    exit 1
fi

# Install the package
echo "Installing package..."
sudo dpkg -i cloudflared-latest.deb

# Fix any dependencies
sudo apt-get install -f -y

# Clean up
rm cloudflared-latest.deb

# Verify installation
INSTALLED_VERSION=$(cloudflared --version | head -n1)
echo "✅ Cloudflared installed successfully: $INSTALLED_VERSION"

# Create scripts directory
mkdir -p ~/scripts

echo ""
echo "Next steps:"
echo "1. Authenticate with Cloudflare: cloudflared tunnel login"
echo "2. Create a tunnel: cloudflared tunnel create training-tunnel"
echo "3. Run setup script: ./scripts/setup-cloudflare-tunnel.sh"
echo ""
echo "For more information, see: cloudflare-raspberry-pi-guide.md"