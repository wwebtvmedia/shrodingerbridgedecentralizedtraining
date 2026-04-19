# 🍓 Raspberry Pi Deployment Guide

[![Platform: Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi-red.svg)]()
[![Status: Production Ready](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)]()

Comprehensive guide for deploying the **Swarm Consolidation Server** on Raspberry Pi hardware.

---

## 📑 Table of Contents
- [🎯 Overview](#-overview)
- [📋 Prerequisites](#-prerequisites)
- [🚀 Quick Setup (Automated)](#-quick-setup-automated)
- [🛠️ Manual Setup Steps](#️-manual-setup-steps)
- [🛡️ Security & Performance](#️-security--performance)
- [❓ Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This guide details how to transform a Raspberry Pi into a central consolidation hub for the Swarm Schrödinger Bridge training system. This hub facilitates model synchronization, data persistence, and global state management for the decentralized browser clients.

---

## 📋 Prerequisites

- **Hardware**: Raspberry Pi 4 (4GB+) or Raspberry Pi 5.
- **OS**: Raspberry Pi OS 64-bit (Debian Bullseye/Bookworm).
- **Environment**: Node.js 18+ and `npm`.
- **Network**: Stable internet connection (Ethernet preferred).

---

## 🚀 Quick Setup (Automated)

We provide a comprehensive setup script that handles system updates, dependency installation, and service configuration.

```bash
# 1. Clone repository
git clone https://github.com/wwebtvmedia/shrodingerbridgedecentralizedtraining.git
cd shrodingerbridgedecentralizedtraining

# 2. Run setup script
chmod +x setup-raspberry-pi.sh
./setup-raspberry-pi.sh
```

---

## 🛠️ Manual Setup Steps

### 1. Dependency Installation
```bash
sudo apt update && sudo apt upgrade -y
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs git
```

### 2. Service Configuration
Create a systemd unit file at `/etc/systemd/system/consolidation-server.service`:

```ini
[Unit]
Description=Swarm Consolidation Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/prototype
Environment=NODE_ENV=production
Environment=PORT=8080
ExecStart=/usr/bin/node server/index.js
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## 🛡️ Security & Performance

### Memory Optimization
For stability on RPi, we recommend increasing the swap size:
1.  Edit `/etc/dphys-swapfile`.
2.  Set `CONF_SWAPSIZE=2048`.
3.  Run `sudo dphys-swapfile setup && sudo dphys-swapfile swapon`.

### Network Security
> [!IMPORTANT]
> **Never** expose port 8080 directly to the internet. Always use the **Cloudflare Tunnel** method described in [cloudflare-raspberry-pi-guide.md](./cloudflare-raspberry-pi-guide.md).

---

## ❓ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Service won't start** | Check logs: `sudo journalctl -u consolidation-server -xe`. |
| **Out of Memory** | Increase swap size or reduce `BATCH_SIZE` in `src/config.js`. |
| **Permission Denied** | Ensure the `pi` user owns the project directory: `chown -R pi:pi .`. |

---
_Last updated: April 19, 2026_
