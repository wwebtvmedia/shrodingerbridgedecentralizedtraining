# Migration Guide: From Hardcoded URLs to Environment Variables

This guide helps you migrate from the old hardcoded `tree4five.com` URLs to the new environment variable-based configuration.

## What Changed

Previously, scripts used hardcoded values:

- `DOMAIN="tree4five.com"`
- `SUBDOMAIN="training"`

Now, all configuration uses environment variables loaded from `.env` file:

- `URL=https://training.yourdomain.com` (primary configuration)
- Scripts automatically parse URL to extract domain and subdomain

## Migration Steps

### Step 1: Create Your .env File

```bash
# Copy the template
cp .env.template .env

# Edit with your values
nano .env
```

### Step 2: Configure Your URL

In the `.env` file, set your URL:

```bash
# For a subdomain like training.yourdomain.com
URL=https://training.yourdomain.com

# For a root domain like yourdomain.com
URL=https://yourdomain.com
# Also set CLOUDFLARE_TUNNEL_SUBDOMAIN=training if using root domain
```

### Step 3: Update Existing Tunnel (If Already Configured)

If you already have a Cloudflare Tunnel configured with the old hardcoded values:

1. **Update the configuration file**:

   ```bash
   sudo nano /etc/cloudflared/config.yml
   ```

   Update all `hostname:` entries from `training.tree4five.com` to your new domain.

2. **Update DNS records** in Cloudflare dashboard:
   - Change CNAME records from `*.tree4five.com` to your new domain
   - Update the target if tunnel ID changed

3. **Restart the service**:
   ```bash
   sudo systemctl restart cloudflared
   ```

### Step 4: Test the Migration

Run the health check to verify everything works:

```bash
./scripts/cloudflare-health-check.sh
```

## Script Compatibility

### Updated Scripts

All scripts in the `scripts/` directory now support environment variables:

1. **`setup-cloudflare-tunnel.sh`** - Reads from `.env`, parses `URL` variable
2. **`cloudflare-health-check.sh`** - Uses environment variables for configuration
3. **`update-cloudflared.sh`** - Maintains backward compatibility
4. **`install-cloudflared.sh`** - Unchanged (no URL dependencies)

### Backward Compatibility

The scripts maintain backward compatibility:

- If `.env` file exists: Uses environment variables
- If `.env` doesn't exist but `.env.template` exists: Shows warning, uses template
- If neither exists: Shows error with instructions

## Environment Variable Reference

### Required Variables

| Variable | Description                | Example                        |
| -------- | -------------------------- | ------------------------------ |
| `URL`    | Your public URL (https://) | `https://training.example.com` |

### Optional Variables

| Variable                       | Default                         | Description                     |
| ------------------------------ | ------------------------------- | ------------------------------- |
| `CLOUDFLARE_TUNNEL_NAME`       | `training-consolidation-tunnel` | Name for your Cloudflare Tunnel |
| `CLOUDFLARE_TUNNEL_SUBDOMAIN`  | `training`                      | Subdomain (if not in URL)       |
| `CLOUDFLARE_TUNNEL_LOCAL_PORT` | `8080`                          | Local server port               |
| `SERVER_PORT`                  | `8080`                          | Server port                     |
| `NODE_ENV`                     | `production`                    | Node.js environment             |

### Derived Variables (Auto-calculated)

The scripts automatically calculate these from the `URL`:

| Derived Value | How it's calculated             | Example                |
| ------------- | ------------------------------- | ---------------------- |
| `FULL_DOMAIN` | URL without `https://` and path | `training.example.com` |
| `DOMAIN`      | Base domain                     | `example.com`          |
| `SUBDOMAIN`   | First part if subdomain exists  | `training`             |

## Common Migration Scenarios

### Scenario 1: Changing Domain Only

You want to change from `tree4five.com` to `yourdomain.com`:

1. In `.env`: `URL=https://training.yourdomain.com`
2. Update DNS records in Cloudflare
3. Restart tunnel: `sudo systemctl restart cloudflared`

### Scenario 2: Changing Subdomain

You want to change from `training` to `ai-training`:

1. In `.env`: `URL=https://ai-training.yourdomain.com`
2. Update `/etc/cloudflared/config.yml` hostnames
3. Update DNS records
4. Restart service

### Scenario 3: Using Root Domain

You want to use root domain without subdomain:

1. In `.env`: `URL=https://yourdomain.com`
2. Also set: `CLOUDFLARE_TUNNEL_SUBDOMAIN=` (empty)
3. Update configuration and DNS accordingly

## Troubleshooting Migration

### Issue: Script Can't Find .env File

```
Error: No .env or .env.template file found.
```

**Solution**:

```bash
cp .env.template .env
# Edit .env with your values
```

### Issue: URL Parsing Incorrect

If your URL isn't parsed correctly (e.g., `https://example.com/path`):

**Solution**: Ensure URL doesn't have paths:

```bash
# Wrong
URL=https://example.com/app

# Correct
URL=https://app.example.com
```

### Issue: DNS Not Updated

After changing domain, you get certificate errors or connection failures.

**Solution**:

1. Update DNS records in Cloudflare dashboard
2. Wait for propagation (up to 24 hours)
3. Check: `dig your-new-domain.com`

### Issue: Tunnel Won't Start After Changes

```
sudo systemctl status cloudflared
# Shows errors
```

**Solution**:

1. Check logs: `sudo journalctl -u cloudflared -xe`
2. Verify config syntax: `sudo cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate`
3. Re-authenticate if needed: `cloudflared tunnel login`

## Verification Checklist

After migration, verify:

- [ ] `.env` file exists with correct `URL`
- [ ] `./scripts/setup-cloudflare-tunnel.sh` loads variables correctly
- [ ] `/etc/cloudflared/config.yml` has updated hostnames
- [ ] DNS records point to correct tunnel
- [ ] `sudo systemctl status cloudflared` shows "active"
- [ ] `curl https://your-new-domain.com/api/health` works
- [ ] `./scripts/cloudflare-health-check.sh` shows no errors

## Rolling Back

If you need to revert to hardcoded values temporarily:

1. Edit scripts and change back to hardcoded values
2. Or create `.env` with old domain: `URL=https://training.tree4five.com`
3. Update DNS records back to old domain

## Support

For issues with migration:

1. Check script output for error messages
2. Review logs: `sudo journalctl -u cloudflared -f`
3. Consult the main guide: `cloudflare-raspberry-pi-guide.md`
4. Check Cloudflare documentation: https://developers.cloudflare.com/cloudflare-one/

## Summary

The migration to environment variables provides:

- **Security**: Sensitive configuration not in git
- **Flexibility**: Easy domain changes without editing scripts
- **Consistency**: Single source of truth for URL configuration
- **Portability**: Easy to deploy to different environments

Update your configuration and enjoy the improved flexibility!
