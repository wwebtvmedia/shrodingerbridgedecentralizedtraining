# Cloudflare Raspberry Pi Scripts

This directory contains scripts to help you set up and manage Cloudflare services on your Raspberry Pi for the Training Consolidation Server.

## Scripts Overview

### 1. `install-cloudflared.sh`
**Purpose**: Install Cloudflared on Raspberry Pi (ARM64/ARMv7)
**Usage**: 
```bash
chmod +x scripts/install-cloudflared.sh
./scripts/install-cloudflared.sh
```

### 2. `setup-cloudflare-tunnel.sh`
**Purpose**: Complete Cloudflare Tunnel setup for Training Consolidation Server
**Usage**:
```bash
chmod +x scripts/setup-cloudflare-tunnel.sh
# Edit the configuration variables at the top of the script first
./scripts/setup-cloudflare-tunnel.sh
```

### 3. `cloudflare-health-check.sh`
**Purpose**: Comprehensive health check for Cloudflare Tunnel
**Usage**:
```bash
chmod +x scripts/cloudflare-health-check.sh
./scripts/cloudflare-health-check.sh
```

### 4. `update-cloudflared.sh`
**Purpose**: Update Cloudflared to the latest version
**Usage**:
```bash
chmod +x scripts/update-cloudflared.sh
./scripts/update-cloudflared.sh
```

## Quick Start Guide

### Step 1: Make Scripts Executable
```bash
chmod +x scripts/*.sh
```

### Step 2: Install Cloudflared
```bash
./scripts/install-cloudflared.sh
```

### Step 3: Configure Environment Variables
1. Copy the template and configure your environment:
```bash
cp .env.template .env
nano .env  # Edit with your values
```

2. Key variables to set in `.env`:
   - `URL=https://training.yourdomain.com` (your public URL)
   - `CLOUDFLARE_TUNNEL_NAME` (optional, default: training-consolidation-tunnel)
   - `CLOUDFLARE_TUNNEL_LOCAL_PORT` (optional, default: 8080)

3. Run the setup:
```bash
./scripts/setup-cloudflare-tunnel.sh
```

**Important**: The `.env` file is not committed to git (see `.gitignore`). The `URL` variable will be parsed to extract domain and subdomain automatically.

### Step 4: Verify Setup
```bash
./scripts/cloudflare-health-check.sh
```

## Configuration Files

### Cloudflared Config (`/etc/cloudflared/config.yml`)
The setup script creates this configuration file with:
- Tunnel connection settings
- Ingress rules for your domain/subdomains
- Logging configuration
- Health check endpoints

### Environment Variables
Consider creating a `.env` file for your server with:
```bash
CLOUDFLARE_TUNNEL_ENABLED=true
PUBLIC_URL=https://training.yourdomain.com
WS_URL=wss://ws.training.yourdomain.com
```

## Maintenance

### Regular Health Checks
Run the health check script periodically:
```bash
# Add to crontab for daily checks
0 8 * * * /home/pi/prototype/scripts/cloudflare-health-check.sh >> /var/log/cloudflare-health.log
```

### Updates
Check for updates monthly:
```bash
./scripts/update-cloudflared.sh
```

### Monitoring Logs
```bash
# Live logs
sudo journalctl -u cloudflared -f

# Recent errors
sudo journalctl -u cloudflared --since "1 hour ago" | grep -i error
```

## Troubleshooting

### Common Issues

1. **Tunnel won't start**
   ```bash
   sudo systemctl restart cloudflared
   sudo journalctl -u cloudflared -xe
   ```

2. **DNS not resolving**
   - Check Cloudflare DNS settings
   - Ensure CNAME records point to correct tunnel ID
   - Wait for DNS propagation (up to 24 hours)

3. **Certificate errors**
   ```bash
   cloudflared tunnel login  # Re-authenticate
   sudo systemctl restart cloudflared
   ```

4. **High memory usage**
   ```bash
   # Check memory
   free -h
   # Restart service
   sudo systemctl restart cloudflared
   ```

### Diagnostic Commands
```bash
# List all tunnels
cloudflared tunnel list

# Get tunnel info
cloudflared tunnel info training-consolidation-tunnel

# Test connectivity
curl https://training.yourdomain.com/api/health

# Validate config
sudo cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate
```

## Security Considerations

1. **Keep Cloudflared Updated**: Regular updates include security patches
2. **Use Cloudflare Access**: Add zero-trust security for sensitive applications
3. **Monitor Logs**: Regularly check for unauthorized access attempts
4. **Backup Configuration**: Backup `/etc/cloudflared/` and `~/.cloudflared/` regularly

## Integration with Training Consolidation Server

### Server Configuration
Update your `server/index.js` to use environment variables:

```javascript
const config = {
  port: process.env.PORT || 8080,
  publicUrl: process.env.PUBLIC_URL || `http://localhost:${process.env.PORT || 8080}`,
  cloudflareTunnel: process.env.CLOUDFLARE_TUNNEL_ENABLED === 'true'
};
```

### Client Configuration
Update client WebSocket connections:

```javascript
// In src/consolidation-client.js
const wsUrl = process.env.WS_URL || `ws://localhost:${window.location.port}`;
```

## Performance Tips

1. **Use Wired Connection**: Ethernet provides more stable tunnel connection
2. **Monitor Resource Usage**: Raspberry Pi has limited RAM (check with `htop`)
3. **Optimize Logging**: Set `loglevel: warn` in production to reduce log volume
4. **Regular Maintenance**: Restart service weekly to clear memory

## Support

For issues:
1. Check the comprehensive guide: `../cloudflare-raspberry-pi-guide.md`
2. Review script output and logs
3. Check Cloudflare status: https://www.cloudflarestatus.com/
4. Consult Cloudflare documentation: https://developers.cloudflare.com/cloudflare-one/

## License

These scripts are provided as part of the Training Consolidation Server project.
See the main project LICENSE for details.