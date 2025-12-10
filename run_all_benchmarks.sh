#!/bin/bash
# run_all_benchmarks.sh - Test all CARTS benchmarks
#
# Usage: ./run_all_benchmarks.sh [size]
#   size: small (default), medium, large

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SIZE="${1:-small}"

# Build counters
PASSED=0
FAILED=0
SKIPPED=0

# Run counters
RUN_PASSED=0
RUN_FAILED=0

# These can be fixed by removing the null checks from the source
SKIP_LIST=""

echo "========================================"
echo "Testing CARTS Benchmarks (size: $SIZE)"
echo "========================================"
echo ""
printf "%-35s | %-7s | %-5s | %s\n" "Benchmark" "Build" "Run" "Note"
printf "%-35s-+-%-7s-+-%-5s-+-%s\n" "-----------------------------------" "-------" "-----" "-----"

# Get benchmarks - take only the first token when it already contains a slash
BENCHMARKS=$(carts benchmarks list 2>/dev/null | awk '$1 ~ /\// {print $1}')

for bench in $BENCHMARKS; do
    # Skip known broken
    if [[ " $SKIP_LIST " =~ " $bench " ]]; then
        printf "%-35s | %-7s | %-5s | %s\n" "$bench" "SKIP" "-" "known issue"
        ((SKIPPED++))
        continue
    fi

    # Build with specified size, capture output
    if OUTPUT=$(carts benchmarks "$bench" "$SIZE" 2>&1); then
        ((PASSED++))

        # Find and execute the binary
        BENCH_DIR="$SCRIPT_DIR/$bench"
        BINARY_NAME="$(basename "$bench")_arts"
        BINARY_PATH="$BENCH_DIR/$BINARY_NAME"

        if [ -f "$BINARY_PATH" ]; then
            # Run with timeout to catch hangs (use gtimeout on macOS)
            TIMEOUT_CMD="timeout"
            if ! command -v timeout &> /dev/null; then
                if command -v gtimeout &> /dev/null; then
                    TIMEOUT_CMD="gtimeout"
                else
                    TIMEOUT_CMD=""
                fi
            fi

            # Execute the binary
            if [ -n "$TIMEOUT_CMD" ]; then
                RUN_OUTPUT=$("$TIMEOUT_CMD" 30s "$BINARY_PATH" 2>&1) && RUN_EXIT=0 || RUN_EXIT=$?
            else
                RUN_OUTPUT=$("$BINARY_PATH" 2>&1) && RUN_EXIT=0 || RUN_EXIT=$?
            fi

            if [ $RUN_EXIT -eq 0 ]; then
                RUN_STATUS="OK"
                ((RUN_PASSED++))
            elif [ $RUN_EXIT -eq 139 ] || [ $RUN_EXIT -eq 134 ] || [ $RUN_EXIT -eq 136 ] || [ $RUN_EXIT -eq 124 ]; then
                # 139=SEGV, 134=ABRT, 136=FPE, 124=timeout
                RUN_STATUS="CRASH"
                ((RUN_FAILED++))
            else
                RUN_STATUS="FAIL"
                ((RUN_FAILED++))
            fi
        else
            RUN_STATUS="-"  # No binary generated
        fi

        printf "%-35s | %-7s | %-5s | %s\n" "$bench" "PASS" "$RUN_STATUS" ""
    else
        # Build failed
        ERROR=$(echo "$OUTPUT" | grep -E "(error:|ERROR)" | head -1 | cut -c1-30)
        printf "%-35s | %-7s | %-5s | %s\n" "$bench" "FAIL" "-" "$ERROR"
        ((FAILED++))
    fi
done

echo ""
echo "========================================"
echo "Build: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "Run:   $RUN_PASSED passed, $RUN_FAILED failed"
echo "========================================"

# Exit with non-zero if any failed
if [ $FAILED -gt 0 ] || [ $RUN_FAILED -gt 0 ]; then
    exit 1
fi
