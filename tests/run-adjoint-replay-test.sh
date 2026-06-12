#!/bin/bash
# Adjoint replay regression test (test levels T2/T3 of the adjoint plan).
#
# 1. Runs a forward simulation with per-substep adjoint recording
#    (--adjoint-save) including the converged residual/Jacobian of every
#    accepted substep (--adjoint-save-system).
# 2. Re-runs the same binary with --adjoint-mode=replay, which walks
#    backwards over all recorded substeps,
#    recreates each converged linearization from the stored snapshots and
#    compares it with the stored system.
#
# The test passes when every substep matches within the replay tolerance.
#
# Usage:
#   run-adjoint-replay-test.sh <flow-binary> <deck> <output-dir> [tolerance]
#
# Both stages use the same binary (e.g. flow_blackoil); the replay stage is
# selected with --adjoint-mode=replay.

set -u

FLOW=${1:?usage: $0 <flow-binary> <deck> <output-dir> [tolerance]}
DECK=${2:?missing deck}
OUTDIR=${3:?missing output dir}
TOL=${4:-1e-12}

# Strict, replay-friendly settings: single thread, no storage cache
# (the replay relies on the cache-free storage term, where the residual
# is a pure function of the current and previous states).
COMMON_ARGS="--threads-per-process=1 --enable-storage-cache=false"

mkdir -p "${OUTDIR}"

echo "=== forward run with adjoint recording"
${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${COMMON_ARGS} \
    --adjoint-save=true --adjoint-save-system=true > "${OUTDIR}/forward.log" 2>&1
status=$?
if [ ${status} -ne 0 ]; then
    echo "FAILURE: forward run exited with status ${status} (log: ${OUTDIR}/forward.log)"
    exit ${status}
fi

echo "=== backward replay"
${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${COMMON_ARGS} \
    --adjoint-mode=replay \
    --adjoint-replay-tolerance="${TOL}" > "${OUTDIR}/replay.log" 2>&1
status=$?

grep -E "replay substep|Adjoint replay finished" "${OUTDIR}/replay.log" | tail -8

if [ ${status} -eq 0 ]; then
    echo "T2/T3 adjoint replay: PASS (tolerance ${TOL})"
else
    echo "T2/T3 adjoint replay: FAIL (see ${OUTDIR}/replay.log)"
fi
exit ${status}
