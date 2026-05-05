#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$PROJECT_DIR")"
COMPOSE_FILE="$REPO_ROOT/docker-compose.yml"

# Aurelius Testnet Deployment Script
# Prerequisites:
#   - Python 3.10+ with venv
#   - Docker and Docker Compose
#   - Bittensor wallet created (btcli wallet new_coldkey / new_hotkey)
#   - ~100 test TAO for subnet registration

echo "=== Aurelius Testnet Deployment ==="

# Configuration
NETWORK="${BT_SUBTENSOR_NETWORK:-test}"
NETUID="${BT_NETUID:-455}"
WALLET_NAME="${WALLET_NAME:-default}"
WALLET_HOTKEY="${WALLET_HOTKEY:-default}"

echo "Network: $NETWORK"
echo "NetUID: $NETUID"
echo "Wallet: $WALLET_NAME / $WALLET_HOTKEY"

# Step 1: Install dependencies
echo ""
echo "--- Step 1: Installing dependencies ---"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -e ".[api,ml,simulation,dev]" -q

# Step 2: Start infrastructure
echo ""
echo "--- Step 2: Starting PostgreSQL + API ---"

# Source .env for credentials if available
if [ -f ".env" ]; then
    source .env 2>/dev/null || true
fi

docker compose -f "$COMPOSE_FILE" --project-directory "$REPO_ROOT" up -d db
echo "Waiting for PostgreSQL..."
for i in $(seq 1 30); do
    if docker compose -f "$COMPOSE_FILE" --project-directory "$REPO_ROOT" exec -T db pg_isready -U "${POSTGRES_USER:-aurelius}" &>/dev/null; then
        echo "PostgreSQL is ready."
        break
    fi
    [ "$i" -eq 30 ] && echo "WARNING: PostgreSQL health check timed out"
    sleep 1
done

# Start API (migrations run automatically via entrypoint)
docker compose -f "$COMPOSE_FILE" --project-directory "$REPO_ROOT" up -d api
echo "Waiting for API..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is healthy"
        break
    fi
    [ "$i" -eq 30 ] && echo "WARNING: API health check failed"
    sleep 1
done

# Step 3: Register on testnet (manual step)
echo ""
echo "--- Step 3: Subnet Registration ---"
echo "If not already registered, run:"
echo "  btcli subnet register --netuid $NETUID --network $NETWORK --wallet.name $WALLET_NAME --wallet.hotkey $WALLET_HOTKEY"
echo ""

# Step 4: Start validator
echo ""
echo "--- Step 4: Ready to start ---"
echo ""
echo "Start validator:"
echo "  BT_SUBTENSOR_NETWORK=$NETWORK BT_NETUID=$NETUID WALLET_NAME=$WALLET_NAME WALLET_HOTKEY=$WALLET_HOTKEY aurelius-validator"
echo ""
echo "Start miner (in another terminal):"
echo "  BT_SUBTENSOR_NETWORK=$NETWORK BT_NETUID=$NETUID WALLET_NAME=$WALLET_NAME WALLET_HOTKEY=miner_hotkey aurelius-miner"
echo ""
echo "=== Deployment ready ==="
