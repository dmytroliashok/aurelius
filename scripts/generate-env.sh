#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"
EXAMPLE_FILE="$PROJECT_DIR/.env.example"

if [ -f "$ENV_FILE" ]; then
    echo ".env already exists at $ENV_FILE"
    echo "Remove it first if you want to regenerate."
    exit 1
fi

if [ ! -f "$EXAMPLE_FILE" ]; then
    echo "ERROR: .env.example not found at $EXAMPLE_FILE"
    exit 1
fi

# Generate secrets
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
ADMIN_API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
POSTGRES_PASSWORD=$(python3 -c "import secrets; print(secrets.token_hex(16))")

# Copy template and fill in secrets
cp "$EXAMPLE_FILE" "$ENV_FILE"

# Uncomment and set JWT_SECRET
sed -i.bak "s|^# JWT_SECRET=.*|JWT_SECRET=$JWT_SECRET|" "$ENV_FILE"
# Uncomment and set ADMIN_API_KEY
sed -i.bak "s|^# ADMIN_API_KEY=.*|ADMIN_API_KEY=$ADMIN_API_KEY|" "$ENV_FILE"
# Set POSTGRES_PASSWORD
sed -i.bak "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$POSTGRES_PASSWORD|" "$ENV_FILE"
# Uncomment DATABASE_URL lines
sed -i.bak "s|^# DATABASE_URL=|DATABASE_URL=|" "$ENV_FILE"
sed -i.bak "s|^# DATABASE_URL_SYNC=|DATABASE_URL_SYNC=|" "$ENV_FILE"

# Clean up sed backup files
rm -f "$ENV_FILE.bak"

echo ""
echo "=== .env generated at $ENV_FILE ==="
echo ""
echo "Auto-generated secrets:"
echo "  JWT_SECRET       (64 hex chars)"
echo "  ADMIN_API_KEY    (64 hex chars)"
echo "  POSTGRES_PASSWORD (32 hex chars)"
echo ""
echo "You still need to set:"
echo "  LLM_API_KEY          — Your OpenAI or Anthropic API key (required for simulation)"
echo "  BT_SUBTENSOR_NETWORK — 'test' or 'finney'"
echo "  WALLET_NAME          — Your Bittensor wallet name"
echo "  WALLET_HOTKEY        — Your Bittensor hotkey name"
echo ""
echo "Edit $ENV_FILE to customize, then run:"
echo "  docker compose up -d    # Start PostgreSQL + API"
echo ""
