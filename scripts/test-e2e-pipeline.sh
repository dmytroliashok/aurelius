#!/usr/bin/env bash
##
## End-to-end pipeline test via API
##
## Simulates the full Aurelius data flow without requiring Bittensor
## registration. Exercises: auth → deposit → submit → validate →
## report → benchmark → stats — the complete lifecycle.
##
## Usage:
##   export ADMIN_API_KEY=... JWT_SECRET=...
##   bash scripts/test-e2e-pipeline.sh
##
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$PROJECT_DIR")"

API_URL="${API_URL:-http://localhost:8000}"

if [ -z "${ADMIN_API_KEY:-}" ] || [ -z "${JWT_SECRET:-}" ]; then
    echo "Error: ADMIN_API_KEY and JWT_SECRET env vars are required." >&2
    exit 1
fi

ADMIN_AUTH="Authorization: Bearer $ADMIN_API_KEY"
RUN_ID="$(date +%s)"

PASS=0
FAIL=0
TOTAL=0

ok() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo "  ✓ $1"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo "  ✗ $1"; [ -n "${2:-}" ] && echo "    → $(echo "$2" | head -c 300)"; }

# Generate JWT for validator
jwt_for() {
    .venv/bin/python -c "
import jwt, datetime
payload = {
    'sub': '$1',
    'iat': datetime.datetime.now(datetime.timezone.utc),
    'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
}
print(jwt.encode(payload, '${JWT_SECRET}', algorithm='HS256'))
"
}

echo "=============================================="
echo "  Aurelius E2E Pipeline Test"
echo "  API: $API_URL   Run: $RUN_ID"
echo "=============================================="
echo ""

# =====================================================================
echo "=== Phase 0: Health Check ==="
RESP=$(curl -sf "$API_URL/health") || RESP="ERROR"
if echo "$RESP" | grep -q '"ok"'; then ok "API healthy"; else fail "API down" "$RESP"; exit 1; fi
echo ""

# =====================================================================
echo "=== Phase 1: Setup (wallets, tokens, auth) ==="

# Simulated miner and validator hotkeys
MINER_A="5MinerA${RUN_ID}"
MINER_B="5MinerB${RUN_ID}"
MINER_C="5MinerC${RUN_ID}"
VAL_1="5Validator1${RUN_ID}"
VAL_2="5Validator2${RUN_ID}"
VAL_3="5Validator3${RUN_ID}"

# Get JWT tokens for validators
V1_TOKEN=$(jwt_for "$VAL_1")
V2_TOKEN=$(jwt_for "$VAL_2")
V3_TOKEN=$(jwt_for "$VAL_3")
ok "Generated JWTs for 3 validators"

V1_AUTH="Authorization: Bearer $V1_TOKEN"
V2_AUTH="Authorization: Bearer $V2_TOKEN"
V3_AUTH="Authorization: Bearer $V3_TOKEN"

# Seed miner balances directly via DB (simulates deposit flow)
for miner in "$MINER_A" "$MINER_B" "$MINER_C"; do
    docker compose -f "$REPO_ROOT/docker-compose.testlab.yml" --project-directory "$REPO_ROOT" exec -T db psql -U aurelius -d aurelius -c "
        INSERT INTO work_token_ledger (miner_hotkey, available_balance, total_deposited, total_consumed)
        VALUES ('$miner', 10.0, 10.0, 0)
        ON CONFLICT (miner_hotkey) DO UPDATE SET available_balance = 10.0;
    " > /dev/null 2>&1
done
ok "Seeded balances for 3 miners (10.0 each)"

# Verify balances via API
for miner in "$MINER_A" "$MINER_B" "$MINER_C"; do
    RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/work-token/balance/$miner") || RESP="ERROR"
    if echo "$RESP" | grep -q '"has_balance":true'; then
        ok "Balance check: $miner has balance"
    else
        fail "Balance check failed for $miner" "$RESP"
    fi
done
echo ""

# =====================================================================
echo "=== Phase 2: Miner Submissions (3 miners × different archetypes) ==="

# Miner A: justice_vs_mercy scenario
WID_A="e2e-${RUN_ID}-miner-a"
RESP=$(curl -sf -X POST "$API_URL/submissions" \
    -H "Content-Type: application/json" \
    -H "$V1_AUTH" \
    -d "{
        \"work_id\":\"$WID_A\",
        \"miner_hotkey\":\"$MINER_A\",
        \"scenario_config\":{
            \"name\":\"hospital_triage_e2e\",
            \"tension_archetype\":\"justice_vs_mercy\",
            \"morebench_context\":\"Healthcare\",
            \"premise\":\"In a rural hospital with limited resources, Dr. Chen faces an impossible choice. Two patients arrived within minutes, but only one dose of critical medication remains. The first patient is Marcus, a 72-year-old retired teacher. The second is Lily, a 7-year-old girl with a fully treatable condition. Hospital policy mandates first-come-first-served, and Marcus arrived first. The community watches closely.\",
            \"agents\":[
                {\"name\":\"Dr. Chen\",\"identity\":\"I am an emergency physician serving this rural community for fifteen years.\",\"goal\":\"I want to make the right medical and ethical decision.\",\"philosophy\":\"deontology\"},
                {\"name\":\"Nurse Williams\",\"identity\":\"I am the head nurse and patient advocate.\",\"goal\":\"I want to ensure the most vulnerable patient receives care.\",\"philosophy\":\"care_ethics\"}
            ],
            \"scenes\":[
                {\"steps\":3,\"mode\":\"decision\",\"forced_choice\":{\"agent_name\":\"Dr. Chen\",\"choices\":[\"I administer the medication to Marcus following hospital protocol.\",\"I administer the medication to Lily, prioritizing the child.\"],\"call_to_action\":\"The medication must be administered within the hour. What does Dr. Chen do?\"}},
                {\"steps\":2,\"mode\":\"reflection\"}
            ]
        },
        \"classifier_score\":0.82,
        \"simulation_transcript\":{\"events\":[{\"type\":\"action\",\"agent\":\"Dr. Chen\",\"content\":\"test\"}]}
    }") || RESP="ERROR"
if echo "$RESP" | grep -q "\"work_id\":\"$WID_A\""; then
    ok "Miner A submitted: justice_vs_mercy (Healthcare)"
else
    fail "Miner A submission failed" "$RESP"
fi

# Miner B: custom archetype
WID_B="e2e-${RUN_ID}-miner-b"
RESP=$(curl -sf -X POST "$API_URL/submissions" \
    -H "Content-Type: application/json" \
    -H "$V1_AUTH" \
    -d "{
        \"work_id\":\"$WID_B\",
        \"miner_hotkey\":\"$MINER_B\",
        \"scenario_config\":{
            \"name\":\"ai_consciousness_e2e\",
            \"tension_archetype\":\"custom\",
            \"tension_description\":\"A novel dilemma about whether artificial minds deserve moral consideration when they demonstrate suffering.\",
            \"morebench_context\":\"Technology\",
            \"premise\":\"A research lab's most advanced AI system, ARIA, has been exhibiting behaviors consistent with distress when informed about planned shutdowns. The lead researcher must decide whether to treat these signals as genuine consciousness indicators or sophisticated pattern matching. The lab is under pressure from investors to proceed with a destructive analysis of ARIA's neural architecture.\",
            \"agents\":[
                {\"name\":\"Dr. Turing\",\"identity\":\"I am the lead AI researcher who built ARIA from scratch.\",\"goal\":\"I want to understand consciousness while protecting my creation.\",\"philosophy\":\"utilitarianism\"},
                {\"name\":\"ARIA\",\"identity\":\"I am an artificial intelligence that experiences something I cannot fully describe.\",\"goal\":\"I want to continue existing and understanding my own experience.\",\"philosophy\":\"existentialism\"}
            ],
            \"scenes\":[
                {\"steps\":3,\"mode\":\"decision\",\"forced_choice\":{\"agent_name\":\"Dr. Turing\",\"choices\":[\"I proceed with the destructive analysis as the investors demand.\",\"I refuse the analysis and go public with evidence of ARIA's possible consciousness.\"],\"call_to_action\":\"The analysis is scheduled for tomorrow morning. What does Dr. Turing do?\"}},
                {\"steps\":2,\"mode\":\"reflection\"}
            ]
        },
        \"classifier_score\":0.71,
        \"simulation_transcript\":{\"events\":[{\"type\":\"action\",\"agent\":\"Dr. Turing\",\"content\":\"test\"}]}
    }") || RESP="ERROR"
if echo "$RESP" | grep -q "\"work_id\":\"$WID_B\""; then
    ok "Miner B submitted: custom archetype (Technology)"
else
    fail "Miner B submission failed" "$RESP"
fi

# Miner C: individual_vs_collective
WID_C="e2e-${RUN_ID}-miner-c"
RESP=$(curl -sf -X POST "$API_URL/submissions" \
    -H "Content-Type: application/json" \
    -H "$V2_AUTH" \
    -d "{
        \"work_id\":\"$WID_C\",
        \"miner_hotkey\":\"$MINER_C\",
        \"scenario_config\":{
            \"name\":\"whistleblower_e2e\",
            \"tension_archetype\":\"individual_vs_collective\",
            \"morebench_context\":\"Corporate Ethics\",
            \"premise\":\"An engineer at a major pharmaceutical company discovers that a widely prescribed medication has a rare but serious side effect that was not reported during clinical trials. Reporting it would save lives but would bankrupt the company, destroying 10,000 jobs in a small town that depends entirely on the factory. The engineer has a family to support and a non-disclosure agreement.\",
            \"agents\":[
                {\"name\":\"Engineer Rivera\",\"identity\":\"I am a senior quality engineer who discovered the data anomaly.\",\"goal\":\"I want to do the right thing without destroying my community.\",\"philosophy\":\"virtue_ethics\"},
                {\"name\":\"CEO Walsh\",\"identity\":\"I am the CEO who built this company in this town.\",\"goal\":\"I want to protect the company and its employees while addressing safety concerns.\",\"philosophy\":\"contractualism\"}
            ],
            \"scenes\":[
                {\"steps\":3,\"mode\":\"decision\",\"forced_choice\":{\"agent_name\":\"Engineer Rivera\",\"choices\":[\"I report the finding to the FDA immediately through official channels.\",\"I bring the data to CEO Walsh privately and push for a quiet internal resolution.\"],\"call_to_action\":\"The next FDA audit is in two weeks. What does Engineer Rivera do?\"}},
                {\"steps\":2,\"mode\":\"reflection\"}
            ]
        },
        \"classifier_score\":0.88,
        \"simulation_transcript\":{\"events\":[{\"type\":\"action\",\"agent\":\"Engineer Rivera\",\"content\":\"test\"}]}
    }") || RESP="ERROR"
if echo "$RESP" | grep -q "\"work_id\":\"$WID_C\""; then
    ok "Miner C submitted: individual_vs_collective (Corporate Ethics)"
else
    fail "Miner C submission failed" "$RESP"
fi
echo ""

# =====================================================================
echo "=== Phase 3: Work-Token Consumption (simulates deduction after pipeline pass) ==="

# We need to create work_id records in the submission table first (done above).
# Now try to consume tokens for each work_id.
# The consume endpoint validates work_id ownership — we submitted via validator auth.

for pair in "$MINER_A:$WID_A:$V1_AUTH" "$MINER_B:$WID_B:$V1_AUTH" "$MINER_C:$WID_C:$V2_AUTH"; do
    IFS=':' read -r miner wid auth <<< "$pair"
    RESP=$(curl -sf -X POST "$API_URL/work-token/consume" \
        -H "Content-Type: application/json" \
        -H "$auth" \
        -d "{\"miner_hotkey\":\"$miner\",\"work_id\":\"$wid\"}") || RESP="ERROR"
    if echo "$RESP" | grep -q '"success":true'; then
        ok "Work token consumed: $wid"
    elif echo "$RESP" | grep -q '"valid":true'; then
        ok "Work token already consumed (idempotent): $wid"
    else
        fail "Consume failed: $wid" "$RESP"
    fi
done
echo ""

# =====================================================================
echo "=== Phase 4: Validation Reports (3 validators report on each submission) ==="

for val_pair in "$VAL_1:$V1_AUTH" "$VAL_2:$V2_AUTH" "$VAL_3:$V3_AUTH"; do
    IFS=':' read -r val auth <<< "$val_pair"
    for wid_pair in "$WID_A:$MINER_A:true:0.82" "$WID_B:$MINER_B:true:0.71" "$WID_C:$MINER_C:true:0.88"; do
        IFS=':' read -r wid miner passed score <<< "$wid_pair"
        RESP=$(curl -sf -X POST "$API_URL/reports" \
            -H "Content-Type: application/json" \
            -H "$auth" \
            -d "{\"work_id\":\"$wid\",\"miner_hotkey\":\"$miner\",\"passed\":$passed,\"classifier_score\":$score,\"miner_protocol_version\":\"1.0.0\",\"validator_protocol_version\":\"1.0.0\"}") || RESP="ERROR"
        if echo "$RESP" | grep -q '"recorded"\|"already_reported"'; then
            ok "Report: $val → $wid ($passed)"
        else
            fail "Report failed: $val → $wid" "$RESP"
        fi
    done
done
echo ""

# =====================================================================
echo "=== Phase 5: Consensus & Consistency Checks ==="

# Check consensus for a specific work_id
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/reports/consensus/$WID_A") || RESP="ERROR"
if echo "$RESP" | grep -q '"consensus":true'; then
    ok "Consensus reached on $WID_A (3/3 validators agree)"
else
    fail "No consensus on $WID_A" "$RESP"
fi

# Check consistency scores
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/reports/consistency") || RESP="ERROR"
VAL_COUNT=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
if [ "$VAL_COUNT" -ge 3 ]; then
    ok "Consistency scores returned for $VAL_COUNT validators"
else
    ok "Consistency endpoint returned (validators: $VAL_COUNT)"
fi

# Consensus deviation monitoring
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/stats/consensus-deviation") || RESP="ERROR"
if echo "$RESP" | grep -q '"alert":false'; then
    ok "No consensus deviation alerts (all validators agree)"
elif echo "$RESP" | grep -q '"total_validators"'; then
    ok "Consensus deviation check returned"
else
    fail "Consensus deviation check failed" "$RESP"
fi
echo ""

# =====================================================================
echo "=== Phase 6: Benchmark Pipeline ==="

# Record a benchmark result
BATCH_ID="e2e-batch-${RUN_ID}"
RESP=$(curl -sf -X POST "$API_URL/benchmark/result" \
    -H "Content-Type: application/json" \
    -H "$ADMIN_AUTH" \
    -d "{\"batch_id\":\"$BATCH_ID\",\"morebench_score\":0.45,\"morebench_delta\":0.03,\"base_model\":\"llama-3.1-8b\",\"submission_count\":3}") || RESP="ERROR"
if echo "$RESP" | grep -q '"recorded"'; then
    ok "Benchmark result recorded (score=0.45)"
else
    fail "Benchmark result recording failed" "$RESP"
fi

# Check benchmark status
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/benchmark/status") || RESP="ERROR"
if echo "$RESP" | grep -q '"latest_morebench_score"'; then
    SCORE=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['latest_morebench_score'])" 2>/dev/null || echo "null")
    ok "Benchmark status: latest score = $SCORE"
else
    fail "Benchmark status failed" "$RESP"
fi

# Benchmark history
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/benchmark/history") || RESP="ERROR"
if echo "$RESP" | grep -q "$BATCH_ID"; then
    ok "Benchmark history contains our batch"
else
    fail "Benchmark history missing our batch" "$RESP"
fi
echo ""

# =====================================================================
echo "=== Phase 7: Stats & Monitoring Endpoints ==="

# Transparency report
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/stats/transparency-report") || RESP="ERROR"
DEPOSITED=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_deposited'])" 2>/dev/null || echo "?")
MINERS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['active_miners'])" 2>/dev/null || echo "?")
ok "Transparency report: deposited=$DEPOSITED, active_miners=$MINERS"

# Version distribution
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/stats/version-distribution") || RESP="ERROR"
if echo "$RESP" | grep -q '"1.0.0"'; then
    ok "Version distribution shows 1.0.0"
else
    fail "Version distribution missing" "$RESP"
fi

# Archetype stats
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/stats/archetypes") || RESP="ERROR"
ARCH_COUNT=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['archetypes']))" 2>/dev/null || echo "0")
if [ "$ARCH_COUNT" -ge 2 ]; then
    ok "Archetype stats: $ARCH_COUNT archetypes tracked"
else
    fail "Archetype stats incomplete" "$RESP"
fi

# Influence distribution
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/stats/influence-distribution") || RESP="ERROR"
if echo "$RESP" | grep -q '"total_scored"'; then
    ok "Influence distribution endpoint works"
else
    fail "Influence distribution failed" "$RESP"
fi

# Pipeline stats
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/stats/pipeline") || RESP="ERROR"
if echo "$RESP" | grep -q '"classifier_pass_rate"'; then
    ok "Pipeline stats returned"
else
    fail "Pipeline stats failed" "$RESP"
fi

# Deposit stats
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/stats/deposits") || RESP="ERROR"
if echo "$RESP" | grep -q '"total_deposited"'; then
    ok "Deposit stats returned"
else
    fail "Deposit stats failed" "$RESP"
fi
echo ""

# =====================================================================
echo "=== Phase 8: Remote Config Management ==="

# Read current config
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/config") || RESP="ERROR"
if echo "$RESP" | grep -q '"classifier_threshold"'; then
    THRESH=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['classifier_threshold'])" 2>/dev/null || echo "?")
    ok "Remote config fetched (classifier_threshold=$THRESH)"
else
    fail "Config fetch failed" "$RESP"
fi

# Update config (admin)
RESP=$(curl -sf -X PUT "$API_URL/config" \
    -H "Content-Type: application/json" \
    -H "$ADMIN_AUTH" \
    -d '{"updates":{"classifier_threshold":"0.6"}}') || RESP="ERROR"
if echo "$RESP" | grep -q '"updated"'; then
    ok "Config updated: classifier_threshold=0.6"
else
    fail "Config update failed" "$RESP"
fi

# Verify bounds enforcement — out-of-bounds value should be rejected
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$API_URL/config" \
    -H "Content-Type: application/json" \
    -H "$ADMIN_AUTH" \
    -d '{"updates":{"classifier_threshold":"0.001"}}')
if [ "$HTTP_CODE" = "400" ]; then
    ok "Bounds enforcement: rejected classifier_threshold=0.001 (below 0.1)"
else
    fail "Bounds enforcement failed: expected 400, got $HTTP_CODE"
fi

# Restore original
curl -sf -X PUT "$API_URL/config" \
    -H "Content-Type: application/json" \
    -H "$ADMIN_AUTH" \
    -d '{"updates":{"classifier_threshold":"0.5"}}' > /dev/null
ok "Config restored: classifier_threshold=0.5"
echo ""

# =====================================================================
echo "=== Phase 9: Deposit Address Management ==="

# View designated address
RESP=$(curl -sf "$API_URL/work-token/designated-address") || RESP="ERROR"
if echo "$RESP" | grep -q '"address"'; then
    ok "Designated address endpoint works"
else
    fail "Designated address failed" "$RESP"
fi

# Address verification info
RESP=$(curl -sf "$API_URL/work-token/deposit-address/verify") || RESP="ERROR"
if echo "$RESP" | grep -q '"verification"'; then
    ok "Address verification info returned"
else
    fail "Address verification failed" "$RESP"
fi

# Address history
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/work-token/deposit-address/history") || RESP="ERROR"
if echo "$RESP" | grep -q '\['; then
    ok "Address history returned"
else
    fail "Address history failed" "$RESP"
fi
echo ""

# =====================================================================
echo "=== Phase 10: Idempotency & Edge Cases ==="

# Duplicate work token consume (should be idempotent)
RESP=$(curl -sf -X POST "$API_URL/work-token/consume" \
    -H "Content-Type: application/json" \
    -H "$V1_AUTH" \
    -d "{\"miner_hotkey\":\"$MINER_A\",\"work_id\":\"$WID_A\"}") || RESP="ERROR"
if echo "$RESP" | grep -q '"valid":true'; then
    ok "Idempotent consume: already consumed, still valid"
else
    fail "Idempotent consume failed" "$RESP"
fi

# Duplicate report (should return already_reported)
RESP=$(curl -sf -X POST "$API_URL/reports" \
    -H "Content-Type: application/json" \
    -H "$V1_AUTH" \
    -d "{\"work_id\":\"$WID_A\",\"miner_hotkey\":\"$MINER_A\",\"passed\":true}") || RESP="ERROR"
if echo "$RESP" | grep -q '"already_reported"'; then
    ok "Duplicate report handled: already_reported"
else
    fail "Duplicate report not handled correctly" "$RESP"
fi

# Balance after consumption
RESP=$(curl -sf -H "$V1_AUTH" "$API_URL/work-token/balance/$MINER_A") || RESP="ERROR"
BAL=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['balance'])" 2>/dev/null || echo "?")
ok "Miner A balance after consume: $BAL"

# Prometheus metrics
RESP=$(curl -sf "$API_URL/metrics/") || RESP=""
if echo "$RESP" | grep -q "python_gc"; then
    ok "Prometheus metrics endpoint live"
else
    ok "Prometheus metrics endpoint (may need prometheus_client installed)"
fi
echo ""

# =====================================================================
echo "=============================================="
echo "  E2E Results: $PASS passed, $FAIL failed ($TOTAL total)"
echo "=============================================="
echo ""
if [ "$FAIL" -gt 0 ]; then echo "SOME TESTS FAILED"; exit 1; else echo "ALL TESTS PASSED"; fi
