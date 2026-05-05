#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$PROJECT_DIR")"
COMPOSE_FILE="$REPO_ROOT/docker-compose.yml"
cd "$PROJECT_DIR"

ROLE="${1:-}"

usage() {
    echo "Usage: $0 <role>"
    echo ""
    echo "Roles:"
    echo "  validator  — Install deps, start DB + API, print validator start command"
    echo "  miner      — Install deps, print miner start command"
    echo "  api        — Install deps, start DB + API only"
    echo ""
    echo "Examples:"
    echo "  $0 validator"
    echo "  $0 miner"
    echo "  $0 api"
    exit 1
}

if [ -z "$ROLE" ]; then
    usage
fi

# --- Prerequisite checks ---
check_prereqs() {
    local missing=()

    if ! python3 --version 2>/dev/null | grep -qE "3\.(1[0-9]|[2-9][0-9])"; then
        missing+=("Python 3.10+")
    fi

    if [ "$ROLE" != "miner" ]; then
        if ! command -v docker &>/dev/null; then
            missing+=("Docker")
        fi
        if ! docker compose version &>/dev/null 2>&1; then
            missing+=("Docker Compose v2")
        fi
    fi

    if ! command -v btcli &>/dev/null; then
        missing+=("btcli (pip install bittensor-cli)")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: Missing prerequisites:"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi
}

check_prereqs

# --- Virtual environment ---
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# --- Install dependencies ---
echo ""
echo "=== Installing dependencies ==="
case "$ROLE" in
    validator)
        pip install -e ".[api,ml,simulation]" -q
        ;;
    miner)
        pip install -e "." -q
        ;;
    api)
        pip install -e ".[api,ml]" -q
        ;;
    *)
        usage
        ;;
esac

# --- Generate .env if needed ---
if [ ! -f ".env" ]; then
    echo ""
    echo "=== No .env found — generating one ==="
    bash "$SCRIPT_DIR/generate-env.sh"
    echo ""
    echo "Review .env and re-run this script when ready."
    exit 0
fi

source .env 2>/dev/null || true

# --- Start infrastructure (API/validator roles) ---
if [ "$ROLE" != "miner" ]; then
    echo ""
    echo "=== Starting PostgreSQL ==="
    docker compose -f "$COMPOSE_FILE" --project-directory "$REPO_ROOT" up -d db

    echo "Waiting for PostgreSQL to be healthy..."
    for i in $(seq 1 30); do
        if docker compose -f "$COMPOSE_FILE" --project-directory "$REPO_ROOT" exec -T db pg_isready -U "${POSTGRES_USER:-aurelius}" &>/dev/null; then
            echo "PostgreSQL is ready."
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "ERROR: PostgreSQL did not become healthy in 30s"
            exit 1
        fi
        sleep 1
    done

    echo ""
    echo "=== Starting Central API ==="
    docker compose -f "$COMPOSE_FILE" --project-directory "$REPO_ROOT" up -d api

    echo "Waiting for API to be healthy..."
    for i in $(seq 1 30); do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            echo "API is healthy at http://localhost:8000"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "WARNING: API health check timed out — check logs with: docker compose -f $COMPOSE_FILE logs api"
        fi
        sleep 1
    done
fi

# --- Print next steps ---
echo ""
echo "=== Ready ==="
echo ""

case "$ROLE" in
    validator)
        echo "Start the validator:"
        echo "  source .venv/bin/activate"
        echo "  aurelius-validator"
        echo ""
        echo "Make sure you have:"
        echo "  - A registered hotkey with validator permit on the subnet"
        echo "  - LLM_API_KEY set in .env (for Concordia simulation)"
        echo "  - Docker running (for simulation containers)"
        ;;
    miner)
        echo "Start the miner:"
        echo "  source .venv/bin/activate"
        echo "  aurelius-miner"
        echo ""
        echo "Make sure you have:"
        echo "  - A registered hotkey on the subnet"
        echo "  - Scenario config JSON files in configs/ directory"
        echo "  - Port ${AXON_PORT:-8091} open and reachable by validators"
        ;;
    api)
        echo "Central API is running at http://localhost:8000"
        echo "Health:  curl http://localhost:8000/health"
        echo "Metrics: curl http://localhost:8000/metrics"
        echo "Logs:    docker compose -f $COMPOSE_FILE logs -f api"
        ;;
esac
echo ""
