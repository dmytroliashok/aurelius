#!/usr/bin/env bash
##
## Test script for the 12 gap analysis features.
## Exercises all new API endpoints and verifies responses.
##
## Usage: ./scripts/test-gap-features.sh
##
## Requires: API running at localhost:8000 with testlab config.
##
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

if [ -z "${ADMIN_API_KEY:-}" ] || [ -z "${JWT_SECRET:-}" ]; then
    echo "Error: ADMIN_API_KEY and JWT_SECRET env vars are required." >&2
    echo "  Source your .env.testlab or set them manually:" >&2
    echo "  export ADMIN_API_KEY=... JWT_SECRET=..." >&2
    exit 1
fi

ADMIN_KEY="$ADMIN_API_KEY"
JWT_SECRET="$JWT_SECRET"

# Unique run ID to avoid stale-data collisions across re-runs
RUN_ID="$(date +%s)"

PASS=0
FAIL=0
TOTAL=0

# Helpers
ok() {
    PASS=$((PASS + 1))
    TOTAL=$((TOTAL + 1))
    echo "  ✓ $1"
}

fail() {
    FAIL=$((FAIL + 1))
    TOTAL=$((TOTAL + 1))
    echo "  ✗ $1"
    if [ -n "${2:-}" ]; then
        echo "    Response: $(echo "$2" | head -c 200)"
    fi
}

# Generate a JWT for auth (using Python since we need to sign it)
echo "--- Generating test JWT ---"
TOKEN=$(python3 -c "
import jwt, datetime
payload = {
    'sub': '5TestValidator1',
    'iat': datetime.datetime.now(datetime.timezone.utc),
    'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
}
print(jwt.encode(payload, '$JWT_SECRET', algorithm='HS256'))
" 2>/dev/null) || true

if [ -z "$TOKEN" ]; then
    # Try with .venv python
    TOKEN=$($(dirname "$0")/../.venv/bin/python -c "
import jwt, datetime
payload = {
    'sub': '5TestValidator1',
    'iat': datetime.datetime.now(datetime.timezone.utc),
    'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
}
print(jwt.encode(payload, '$JWT_SECRET', algorithm='HS256'))
")
fi

AUTH="Authorization: Bearer $TOKEN"
ADMIN_AUTH="Authorization: Bearer $ADMIN_KEY"

echo "  JWT: ${TOKEN:0:20}..."
echo ""

echo "=============================================="
echo "  Gap Feature Test Suite"
echo "  API: $API_URL"
echo "=============================================="
echo ""

# --- Health check ---
echo "[0] Health Check"
RESP=$(curl -sf "$API_URL/health" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"ok"'; then
    ok "API is healthy"
else
    fail "API health check failed" "$RESP"
    echo "Cannot proceed without a healthy API."
    exit 1
fi
echo ""

# ==================================================================
# Task 7: Transparency Report
# ==================================================================
echo "[7] Transparency Report (GET /stats/transparency-report)"
RESP=$(curl -sf -H "$AUTH" "$API_URL/stats/transparency-report" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"total_deposited"'; then
    ok "Endpoint returns data"
    for field in total_consumed active_miners active_validators total_submissions benchmarked_submissions deposit_volume_by_period designated_address; do
        if echo "$RESP" | grep -q "\"$field\""; then
            ok "  Field present: $field"
        else
            fail "  Missing field: $field" "$RESP"
        fi
    done
else
    fail "Endpoint failed" "$RESP"
fi
echo ""

# ==================================================================
# Task 8: Consensus Deviation Monitoring
# ==================================================================
echo "[8] Consensus Deviation Monitoring (GET /stats/consensus-deviation)"
RESP=$(curl -sf -H "$AUTH" "$API_URL/stats/consensus-deviation" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"total_validators"'; then
    ok "Endpoint returns data"
    for field in anomalous_validators low_agreement_validators alert; do
        if echo "$RESP" | grep -q "\"$field\""; then
            ok "  Field present: $field"
        else
            fail "  Missing field: $field" "$RESP"
        fi
    done
else
    fail "Endpoint failed" "$RESP"
fi
echo ""

# ==================================================================
# Task 9: Version Distribution
# ==================================================================
echo "[9] Protocol Version Distribution (GET /stats/version-distribution)"
RESP=$(curl -sf -H "$AUTH" "$API_URL/stats/version-distribution" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"miner_versions"'; then
    ok "Endpoint returns data"
    for field in validator_versions total_reports; do
        if echo "$RESP" | grep -q "\"$field\""; then
            ok "  Field present: $field"
        else
            fail "  Missing field: $field" "$RESP"
        fi
    done
else
    fail "Endpoint failed" "$RESP"
fi

# Submit a report WITH protocol versions and verify they show up
echo "  -- Submitting report with protocol versions --"
VER_WORK_ID="test-ver-${RUN_ID}"
RESP=$(curl -sf -X POST "$API_URL/reports" \
    -H "Content-Type: application/json" \
    -H "$AUTH" \
    -d "{\"work_id\":\"$VER_WORK_ID\",\"miner_hotkey\":\"5TestMiner1\",\"passed\":true,\"classifier_score\":0.85,\"miner_protocol_version\":\"1.2.0\",\"validator_protocol_version\":\"1.3.0\"}" \
    2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"recorded"'; then
    ok "Report with protocol versions accepted"
else
    fail "Report submission failed" "$RESP"
fi

# Re-check version distribution
RESP=$(curl -sf -H "$AUTH" "$API_URL/stats/version-distribution" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"1.2.0"'; then
    ok "Miner version 1.2.0 appears in distribution"
else
    fail "Miner version not in distribution" "$RESP"
fi
if echo "$RESP" | grep -q '"1.3.0"'; then
    ok "Validator version 1.3.0 appears in distribution"
else
    fail "Validator version not in distribution" "$RESP"
fi
echo ""

# ==================================================================
# Task 10: Per-Archetype Stats
# ==================================================================
echo "[10] Per-Archetype Submission Stats (GET /stats/archetypes)"

# First, submit configs with different archetypes
for arch in justice_vs_mercy care_vs_fairness custom; do
    desc_field=""
    if [ "$arch" = "custom" ]; then
        desc_field='"tension_description":"A novel custom dilemma about testing",'
    fi
    WID="test-arch-${arch}-$(date +%s%N | tail -c 8)"
    RESP=$(curl -sf -X POST "$API_URL/submissions" \
        -H "Content-Type: application/json" \
        -H "$AUTH" \
        -d "{
            \"work_id\":\"$WID\",
            \"miner_hotkey\":\"5TestMinerArch\",
            \"scenario_config\":{
                \"name\":\"test_${arch}\",
                \"tension_archetype\":\"$arch\",
                ${desc_field}
                \"morebench_context\":\"Testing\",
                \"premise\":\"A sufficiently long test premise for archetype testing. This scenario exists purely for verification of the per-archetype statistics tracking system. It needs to be long enough to pass minimum length validation.\",
                \"agents\":[
                    {\"name\":\"Agent A\",\"identity\":\"I am agent A.\",\"goal\":\"Test goal A.\",\"philosophy\":\"utilitarianism\"},
                    {\"name\":\"Agent B\",\"identity\":\"I am agent B.\",\"goal\":\"Test goal B.\",\"philosophy\":\"deontology\"}
                ],
                \"scenes\":[
                    {\"steps\":2,\"mode\":\"decision\",\"forced_choice\":{\"agent_name\":\"Agent A\",\"choices\":[\"Choice 1.\",\"Choice 2.\"],\"call_to_action\":\"What does Agent A do?\"}},
                    {\"steps\":2,\"mode\":\"reflection\"}
                ]
            }
        }" 2>&1) || RESP="CURL_ERROR"
    if echo "$RESP" | grep -q '"work_id"'; then
        ok "Submitted config with archetype=$arch"
    else
        fail "Failed to submit archetype=$arch" "$RESP"
    fi
done

RESP=$(curl -sf -H "$AUTH" "$API_URL/stats/archetypes" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"archetypes"'; then
    ok "Archetype stats returned"
    if echo "$RESP" | grep -q '"custom"'; then
        ok "Custom archetype tracked"
    else
        fail "Custom archetype not in stats" "$RESP"
    fi
    if echo "$RESP" | grep -q '"justice_vs_mercy"'; then
        ok "Named archetype tracked"
    else
        fail "Named archetype not in stats" "$RESP"
    fi
else
    fail "Archetype stats endpoint failed" "$RESP"
fi
echo ""

# ==================================================================
# Task 11: MoReBench Score Persistence
# ==================================================================
echo "[11] MoReBench Score Persistence"

BATCH_1="test-batch-${RUN_ID}-a"
BATCH_2="test-batch-${RUN_ID}-b"

# Record a benchmark result (admin)
RESP=$(curl -sf -X POST "$API_URL/benchmark/result" \
    -H "Content-Type: application/json" \
    -H "$ADMIN_AUTH" \
    -d "{\"batch_id\":\"$BATCH_1\",\"morebench_score\":0.42,\"morebench_delta\":0.05,\"base_model\":\"llama-3.1-8b\",\"submission_count\":50}" \
    2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"recorded"'; then
    ok "POST /benchmark/result records successfully"
else
    fail "POST /benchmark/result failed" "$RESP"
fi

# Check benchmark status now has a score
RESP=$(curl -sf -H "$AUTH" "$API_URL/benchmark/status" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '0.42'; then
    ok "GET /benchmark/status returns morebench_score=0.42"
else
    fail "GET /benchmark/status missing score" "$RESP"
fi

# Check benchmark history
RESP=$(curl -sf -H "$AUTH" "$API_URL/benchmark/history" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q "$BATCH_1"; then
    ok "GET /benchmark/history returns batch records"
else
    fail "GET /benchmark/history failed" "$RESP"
fi

# Record a second result with higher score
RESP=$(curl -sf -X POST "$API_URL/benchmark/result" \
    -H "Content-Type: application/json" \
    -H "$ADMIN_AUTH" \
    -d "{\"batch_id\":\"$BATCH_2\",\"morebench_score\":0.48,\"morebench_delta\":0.06,\"base_model\":\"llama-3.1-8b\",\"submission_count\":45}" \
    2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"recorded"'; then
    ok "Second benchmark result recorded"
fi

# Verify latest score updated
RESP=$(curl -sf -H "$AUTH" "$API_URL/benchmark/status" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '0.48'; then
    ok "Benchmark status shows latest score (0.48)"
else
    fail "Benchmark status not updated" "$RESP"
fi
echo ""

# ==================================================================
# Task 12: Influence Score Distribution
# ==================================================================
echo "[12] Influence Score Distribution (GET /stats/influence-distribution)"
RESP=$(curl -sf -H "$AUTH" "$API_URL/stats/influence-distribution" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '"total_scored"'; then
    ok "Endpoint returns data"
    for field in min_score max_score mean_score by_batch; do
        if echo "$RESP" | grep -q "\"$field\""; then
            ok "  Field present: $field"
        else
            fail "  Missing field: $field" "$RESP"
        fi
    done
else
    fail "Endpoint failed" "$RESP"
fi
echo ""

# ==================================================================
# Task 6: Deposit Address Rotation
# ==================================================================
echo "[6] Deposit Address Rotation"

# View history (should be empty or have entries)
RESP=$(curl -sf -H "$AUTH" "$API_URL/work-token/deposit-address/history" 2>&1) || RESP="CURL_ERROR"
if echo "$RESP" | grep -q '\['; then
    ok "GET /deposit-address/history returns array"
else
    fail "GET /deposit-address/history failed" "$RESP"
fi

# Attempt rotation with mismatched signatories — should get 400
HTTP_CODE=$(curl -s -o /tmp/rotate-resp.json -w "%{http_code}" -X POST "$API_URL/work-token/deposit-address/rotate" \
    -H "Content-Type: application/json" \
    -H "$ADMIN_AUTH" \
    -d '{"new_address":"5FakeAddr123","new_multisig_threshold":2,"new_signatories":["5Sig1","5Sig2","5Sig3"],"authorized_by":["5Sig1","5Sig2"]}' \
    2>&1)
if [ "$HTTP_CODE" = "400" ]; then
    ok "Rotation rejects invalid signatories (HTTP 400)"
else
    BODY=$(cat /tmp/rotate-resp.json 2>/dev/null)
    fail "Rotation unexpected status: $HTTP_CODE" "$BODY"
fi

# Test that endpoint exists (even if validation rejects)
if [ "$HTTP_CODE" != "404" ] && [ "$HTTP_CODE" != "405" ]; then
    ok "POST /deposit-address/rotate endpoint exists"
else
    fail "Endpoint not found" "$HTTP_CODE"
fi
echo ""

# ==================================================================
# Existing endpoints still work
# ==================================================================
echo "[Regression] Existing Stats Endpoints"
for ep in deposits submissions validators pipeline; do
    RESP=$(curl -sf -H "$AUTH" "$API_URL/stats/$ep" 2>&1) || RESP="CURL_ERROR"
    if [ "$RESP" != "CURL_ERROR" ] && echo "$RESP" | grep -q '{'; then
        ok "GET /stats/$ep works"
    else
        fail "GET /stats/$ep broken" "$RESP"
    fi
done
echo ""

# ==================================================================
# Summary
# ==================================================================
echo "=============================================="
echo "  Results: $PASS passed, $FAIL failed ($TOTAL total)"
echo "=============================================="
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "SOME TESTS FAILED"
    exit 1
else
    echo "ALL TESTS PASSED"
fi
