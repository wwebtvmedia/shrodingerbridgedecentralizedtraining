# 🛠️ Cloudflare Raspberry Pi Scripts

[![Platform: Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi-red.svg)]()
[![Service: Cloudflare](https://img.shields.io/badge/Service-Cloudflare-orange.svg)]()

Automation scripts for managing **Cloudflare Tunnels** on Raspberry Pi for the Training Consolidation Server.

---

## 📂 Scripts Directory

| Script                       | Purpose                              | Usage                                  |
| :--------------------------- | :----------------------------------- | :------------------------------------- |
| `install-cloudflared.sh`     | Installs the `cloudflared` binary.   | `./scripts/install-cloudflared.sh`     |
| `setup-cloudflare-tunnel.sh` | Full tunnel & service configuration. | `./scripts/setup-cloudflare-tunnel.sh` |
| `cloudflare-health-check.sh` | End-to-end connectivity diagnostic.  | `./scripts/cloudflare-health-check.sh` |
| `update-cloudflared.sh`      | Updates to the latest version.       | `./scripts/update-cloudflared.sh`      |

---

## 🚀 Quick Start Guide

### 1. Make Executable

```bash
chmod +x scripts/*.sh
```

### 2. Install Dependencies

```bash
./scripts/install-cloudflared.sh
```

### 3. Run Automated Setup

> [!IMPORTANT]
> Ensure your `.env` file is configured with your domain before running setup.

```bash
./scripts/setup-cloudflare-tunnel.sh
```

---

## 📊 Maintenance & Monitoring

### Regular Health Checks

Add this to your `crontab` for daily health reports:

```bash
0 8 * * * /home/pi/prototype/scripts/cloudflare-health-check.sh >> /var/log/cloudflare-health.log
```

### Live Log Monitoring

```bash
sudo journalctl -u cloudflared -f
```

---

## ⚠️ Troubleshooting Tips

- **Tunnel Offline**: Run `cloudflared tunnel list` to check status.
- **DNS Issues**: Verify CNAME records in the Cloudflare Dashboard.
- **Permission Denied**: Ensure you run service-related commands with `sudo`.

---

_Last updated: April 19, 2026_
