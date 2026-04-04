# Raspberry Pi Deployment Guide

This guide explains how to deploy the Training Consolidation Server on a Raspberry Pi, making it accessible via your domain `tree4five.com` on port 8080.

## Prerequisites

- Raspberry Pi (3B+ or newer recommended)
- Raspberry Pi OS (Bullseye or newer)
- Node.js 18+ installed
- Git installed

## Quick Setup Script

Create a setup script on your Raspberry Pi:

```bash
#!/bin/bash
# setup-consolidation-server.sh

echo "🚀 Setting up Training Consolidation Server on Raspberry Pi"

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Node.js 18 if not present
if ! command -v node &> /dev/null; then
    echo "📥 Installing Node.js 18..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt install -y nodejs
fi

# Install Git if not present
if ! command -v git &> /dev/null; then
    echo "📥 Installing Git..."
    sudo apt install -y git
fi

# Clone the project from GitHub
echo "📁 Setting up project directory..."
PROJECT_DIR="$HOME/training-consolidation"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "📥 Cloning Schrödinger Bridge Decentralized Training repository..."
git clone https://github.com/wwebtvmedia/shrodingerbridgedecentralizedtraining.git .

echo "📦 Installing dependencies..."
npm install

echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/consolidation-server.service > /dev/null <<EOF
[Unit]
Description=Training Consolidation Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=NODE_ENV=production
Environment=PORT=8080
ExecStart=/usr/bin/node server/index.js
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "🚀 Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable consolidation-server
sudo systemctl start consolidation-server

echo "✅ Setup complete!"
echo "📊 Check service status: sudo systemctl status consolidation-server"
echo "🌐 Access web interface: http://$(hostname -I | awk '{print $1}'):8080"
echo "📡 WebSocket: ws://$(hostname -I | awk '{print $1}'):8080"
```

## Manual Setup Steps

### 1. Install Dependencies

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node --version
npm --version
```

### 2. Deploy the Application

```bash
# Create project directory
mkdir -p ~/training-consolidation
cd ~/training-consolidation

# Copy all project files to this directory
# (Assuming you have the files from this prototype)

# Install dependencies
npm install
```

### 3. Configure Environment

Create a `.env` file:

```bash
cat > .env <<EOF
PORT=8080
NODE_ENV=production
MODELS_DIR=./models
LOG_LEVEL=info
EOF
```

### 4. Create Startup Script

Create a startup script `start-server.sh`:

```bash
#!/bin/bash
cd "$(dirname "$0")"
node server/index.js
```

Make it executable:
```bash
chmod +x start-server.sh
```

### 5. Configure as System Service (Recommended)

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/consolidation-server.service
```

Add the following content:

```ini
[Unit]
Description=Training Consolidation Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/training-consolidation
Environment=NODE_ENV=production
Environment=PORT=8080
ExecStart=/usr/bin/node server/index.js
Restart=on-failure
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=consolidation-server

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable consolidation-server
sudo systemctl start consolidation-server

# Check status
sudo systemctl status consolidation-server

# View logs
sudo journalctl -u consolidation-server -f
```

### 6. Configure Firewall (Optional)

If you have a firewall enabled:

```bash
sudo ufw allow 8080/tcp
sudo ufw reload
```

### 7. Configure Static IP (Recommended for Server)

Edit network configuration:

```bash
sudo nano /etc/dhcpcd.conf
```

Add at the end (adjust for your network):

```
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

### 8. Enable Auto-start on Boot

```bash
sudo systemctl enable consolidation-server
```

## Network Configuration

### For Local Network Access

The server will be accessible at:
- Web Interface: `http://<raspberry-pi-ip>:8080`
- WebSocket: `ws://<raspberry-pi-ip>:8080`
- API: `http://<raspberry-pi-ip>:8080/api`

Find your Raspberry Pi's IP address:
```bash
hostname -I
```

### For Internet Access (Port Forwarding)

If you want to access from outside your local network:

1. Configure your router to forward port 8080 to your Raspberry Pi's IP address
2. Consider using a dynamic DNS service if you don't have a static IP
3. **Important**: Add authentication/security measures for production use

## Monitoring and Maintenance

### Check Service Status

```bash
sudo systemctl status consolidation-server
```

### View Logs

```bash
# Follow logs in real-time
sudo journalctl -u consolidation-server -f

# View last 100 lines
sudo journalctl -u consolidation-server -n 100

# View logs from today
sudo journalctl -u consolidation-server --since today
```

### Restart Service

```bash
sudo systemctl restart consolidation-server
```

### Stop Service

```bash
sudo systemctl stop consolidation-server
```

### Update Application

```bash
cd ~/training-consolidation
# Update your code
sudo systemctl restart consolidation-server
```

## Performance Considerations for Raspberry Pi

### Memory Optimization

Raspberry Pi models have limited RAM. To optimize:

1. **Reduce Node.js memory usage**:
   ```bash
   # Add to start command
   node --max-old-space-size=512 server/index.js
   ```

2. **Use swap space** (if not already configured):
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # Set CONF_SWAPSIZE=1024
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

3. **Monitor memory usage**:
   ```bash
   free -h
   htop
   ```

### Storage Considerations

Model files (`latest.pt` and historical models) can be large:

1. **Regular cleanup**: Configure automatic cleanup of old models
2. **External storage**: Store models on USB drive if SD card space is limited
3. **Monitor disk usage**:
   ```bash
   df -h
   du -sh ~/training-consolidation/models/
   ```

### Network Optimization

For multiple clients:

1. **Use wired Ethernet** for better reliability
2. **Monitor network traffic**:
   ```bash
   sudo apt install nethogs
   sudo nethogs
   ```

## Security Considerations

### Basic Security Measures

1. **Change default password** for Raspberry Pi user
2. **Enable firewall**:
   ```bash
   sudo ufw enable
   sudo ufw allow 22/tcp  # SSH
   sudo ufw allow 8080/tcp # Consolidation server
   ```

3. **Use HTTPS** in production:
   - Obtain SSL certificate (Let's Encrypt)
   - Configure reverse proxy (nginx) with SSL termination

### Authentication (For Production)

Add authentication to the server:

1. Modify `server/index.js` to require API keys or tokens
2. Implement user authentication for web interface
3. Use environment variables for secrets

## Troubleshooting

### Service Won't Start

Check for errors:
```bash
sudo journalctl -u consolidation-server -xe
```

Common issues:
- Port 8080 already in use: `sudo lsof -i :8080`
- Permission issues: Check file ownership in project directory
- Node.js version too old: Update to Node.js 18+

### Can't Connect to Server

1. Check if service is running:
   ```bash
   sudo systemctl status consolidation-server
   ```

2. Check if port is listening:
   ```bash
   sudo netstat -tlnp | grep :8080
   ```

3. Check firewall rules:
   ```bash
   sudo ufw status
   ```

4. Test locally on Raspberry Pi:
   ```bash
   curl http://localhost:8080/api/health
   ```

### High Memory Usage

1. Check memory usage:
   ```bash
   free -h
   top
   ```

2. Restart service to clear memory:
   ```bash
   sudo systemctl restart consolidation-server
   ```

3. Consider reducing model cache size in server configuration

## Backup and Recovery

### Backup Configuration

```bash
# Backup important files
BACKUP_DIR="/home/pi/consolidation-backup-$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup models
cp -r ~/training-consolidation/models "$BACKUP_DIR/"

# Backup configuration
cp ~/training-consolidation/.env "$BACKUP_DIR/"
cp ~/training-consolidation/package.json "$BACKUP_DIR/"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
```

### Recovery

To restore from backup:

```bash
# Extract backup
tar -xzf consolidation-backup-YYYYMMDD.tar.gz

# Restore files
cp -r backup/models ~/training-consolidation/
cp backup/.env ~/training-consolidation/

# Restart service
sudo systemctl restart consolidation-server
```

## Scaling for Multiple Raspberry Pis

For larger deployments, consider:

1. **Load balancing**: Use multiple Raspberry Pis with a load balancer
2. **Database**: Use external database for model metadata
3. **Shared storage**: Use NFS or similar for shared model storage

## Exposing with Cloudflare Tunnel

To make your server publicly accessible via your domain `tree4five.com` without opening ports on your router, you can use Cloudflare Tunnel (formerly Argo Tunnel). This creates a secure outbound connection to Cloudflare's network.

### 1. Install Cloudflare Tunnel

```bash
# Download and install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared-linux-arm64.deb
cloudflared --version
```

### 2. Authenticate with Cloudflare

```bash
cloudflared tunnel login
```

This will open a browser (or provide a URL) to log in to your Cloudflare account and authorize the tunnel.

### 3. Create a Tunnel

```bash
cloudflared tunnel create tree4five-tunnel
```

Note the tunnel UUID displayed. Configure the tunnel:

```bash
sudo mkdir -p /etc/cloudflared
sudo nano /etc/cloudflared/config.yml
```

Add the following configuration (replace `tunnel-uuid` with your actual tunnel ID):

```yaml
tunnel: tree4five-tunnel
credentials-file: /home/pi/.cloudflared/tunnel-uuid.json

ingress:
  - hostname: tree4five.com
    service: http://localhost:8080
  - service: http_status:404
```

### 4. Run the Tunnel as a Service

```bash
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

### 5. Configure DNS

In your Cloudflare dashboard, go to DNS settings and create a CNAME record pointing `tree4five.com` to `<tunnel-uuid>.cfargotunnel.com`.

### 6. Verify

Check tunnel status:

```bash
cloudflared tunnel list
cloudflared tunnel info tree4five-tunnel
```

Your server will now be accessible at `https://tree4five.com` via Cloudflare's network.

## Support

For issues or questions:
1. Check logs: `sudo journalctl -u consolidation-server -f`
2. Review this documentation
3. Check project README for additional information
4. Visit project website: https://tree4five.com

---

**Note**: This is a prototype system. For production use, implement additional security measures, monitoring, and backup strategies.