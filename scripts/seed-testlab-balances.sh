#!/usr/bin/env bash
##
## Seed work-token balances for all testlab miners.
## Run after the API and DB containers are up:
##
##   ./scripts/seed-testlab-balances.sh
##
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$PROJECT_DIR")"

WALLET_PATH="${1:-$REPO_ROOT/testlab/wallets}"
COMPOSE_FILE="$REPO_ROOT/docker-compose.testlab.yml"
SEED_AMOUNT="1.0"

MINERS=("miner-0" "miner-1" "miner-2" "miner-3")

echo "=== Seeding testlab work-token balances ==="

for name in "${MINERS[@]}"; do
    hotkey_file="$WALLET_PATH/$name/hotkeys/default"
    if [ ! -f "$hotkey_file" ]; then
        echo "  SKIP $name — hotkey file not found at $hotkey_file"
        continue
    fi

    hotkey=$(python3 -c "
import json
with open('$hotkey_file') as f:
    data = json.load(f)
print(data['ss58Address'])
")

    echo "  $name  hotkey=$hotkey  amount=$SEED_AMOUNT"

    docker compose -f "$COMPOSE_FILE" --project-directory "$REPO_ROOT" exec -T db psql -U aurelius -d aurelius -c "
        INSERT INTO work_token_ledger (miner_hotkey, available_balance, total_deposited, total_consumed)
        VALUES ('$hotkey', $SEED_AMOUNT, $SEED_AMOUNT, 0)
        ON CONFLICT (miner_hotkey)
        DO UPDATE SET available_balance = work_token_ledger.available_balance + $SEED_AMOUNT,
                      total_deposited = work_token_ledger.total_deposited + $SEED_AMOUNT;
    "
done

echo ""
echo "=== Done. Verify with: ==="
echo "  docker compose -f $COMPOSE_FILE exec db psql -U aurelius -d aurelius -c 'SELECT miner_hotkey, available_balance FROM work_token_ledger;'"
