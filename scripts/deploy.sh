#!/usr/bin/env bash
# Deploy latest code to the Pi.
# Run from the repo root: scripts/deploy.sh
# Add to ~/.bashrc: alias sbg-deploy='cd ~/squirrelbgone && scripts/deploy.sh'

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEMD_DIR="/etc/systemd/system"
SERVICES=(squirrelbgone-api squirrelbgone-detect)

cd "$REPO_DIR"

echo "==> Pulling latest code..."
git pull

# Copy any changed service files and reload if needed
needs_reload=0
for svc in "${SERVICES[@]}"; do
    src="$REPO_DIR/systemd/$svc.service"
    dst="$SYSTEMD_DIR/$svc.service"
    if [ ! -f "$src" ]; then
        continue
    fi
    if [ ! -f "$dst" ] || ! diff -q "$src" "$dst" > /dev/null 2>&1; then
        echo "==> Updating $svc.service..."
        sudo cp "$src" "$dst"
        needs_reload=1
    fi
done

if [ "$needs_reload" -eq 1 ]; then
    echo "==> Reloading systemd daemon..."
    sudo systemctl daemon-reload
fi

echo "==> Restarting services..."
sudo systemctl restart "${SERVICES[@]}"

echo "==> Done. Service status:"
systemctl is-active "${SERVICES[@]}"
