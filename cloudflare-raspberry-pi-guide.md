# Cloudflare Integration Guide for Raspberry Pi

[![Platform: Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi-red.svg)]()
[![Service: Cloudflare](https://img.shields.io/badge/Service-Cloudflare-orange.svg)]()

This guide provides comprehensive instructions for deploying the **Training Consolidation Server** on a Raspberry Pi using Cloudflare Tunnels for secure, port-less access.

---

## 📑 Table of Contents
- [🎯 Overview](#-overview)
- [📋 Prerequisites](#-prerequisites)
- [⚙️ Environment Configuration](#️-environment-configuration)
- [🛡️ 1. Cloudflare Tunnel Setup](#️-1-cloudflare-tunnel-setup)
- [🚀 2. Complete Setup Script](#-2-complete-setup-script)
- [📊 3. Monitoring & Maintenance](#-3-monitoring--maintenance)
- [🔐 4. Security Enhancements](#-4-security-enhancements)
- [❓ 7. Troubleshooting](#-7-troubleshooting)

---

## 🎯 Overview

Cloudflare Tunnels (formerly Argo Tunnel) allow you to expose your local Raspberry Pi server to the internet without opening any ports on your router. This is the **recommended** way to deploy the swarm consolidation server.

### Benefits
- **No Port Forwarding**: Protects your home network.
- **DDoS Protection**: Leverages Cloudflare's global network.
- **Automatic SSL**: Free HTTPS certificates managed by Cloudflare.
- **Static URL**: Access your Pi via a clean domain (e.g., `training.yourdomain.com`).

---

## 📋 Prerequisites

- **Hardware**: Raspberry Pi (ARM64 recommended).
- **Software**: Node.js 18+ and `npm`.
- **Account**: A Cloudflare account with a registered domain.
- **Port**: Training Consolidation Server running on port `8080`.

---

## ⚙️ Environment Configuration

1.  **Initialize Environment**:
    ```bash
    cp .env.template .env
    ```
2.  **Configure `.env`**:
    ```bash
    # Core URL
    URL=https://training.yourdomain.com
    
    # Server Settings
    SERVER_PORT=8080
    NODE_ENV=production
    ```

---

## 🛡️ 1. Cloudflare Tunnel Setup

### Step 1: Install Cloudflared
```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared-linux-arm64.deb
```

### Step 2: Authenticate
```bash
cloudflared tunnel login
```

### Step 3: Create & Route
```bash
cloudflared tunnel create training-swarm-tunnel
```

---

## 🚀 2. Complete Setup Script

Use the provided automated scripts for a seamless experience:

```bash
# Install and configure everything
chmod +x scripts/setup-cloudflare-tunnel.sh
./scripts/setup-cloudflare-tunnel.sh
```

---

## 📊 3. Monitoring & Maintenance

| Tool | Command | Description |
| :--- | :--- | :--- |
| **Status** | `systemctl status cloudflared` | Check service health. |
| **Logs** | `journalctl -u cloudflared -f` | View real-time tunnel logs. |
| **Health** | `./scripts/cloudflare-health-check.sh` | Verify end-to-end connectivity. |
| **Update** | `./scripts/update-cloudflared.sh` | Update to the latest version. |

---

## 🔐 4. Security Enhancements

### Zero Trust (Access)
We recommend enabling **Cloudflare Access** to require authentication (e.g., GitHub, Google, or Email OTP) before anyone can access your training dashboard.

1.  Go to **Access -> Applications** in Cloudflare Dashboard.
2.  Add `training.yourdomain.com`.
3.  Set a policy to allow only your email or authorized team members.

---

## ❓ 7. Troubleshooting

> [!TIP]
> Most issues on Raspberry Pi are related to outdated versions or missing credentials.

| Issue | Solution |
| :--- | :--- |
| **Tunnel won't start** | Run `cloudflared tunnel cleanup` and re-login. |
| **DNS not resolving** | Ensure the "Proxied" (orange cloud) status is ON in Cloudflare DNS. |
| **High Memory Usage** | Restart service: `sudo systemctl restart cloudflared`. |
| **404 Errors** | Verify the `ingress` rules in `/etc/cloudflared/config.yml`. |

---
_Last updated: April 19, 2026_
