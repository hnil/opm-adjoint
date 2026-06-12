#!/bin/bash
# Storage-path equivalence test (test level T0.7 of the adjoint plan).
#
# Runs the same deck twice, with --enable-storage-cache=true and =false,
# and compares the results. Expected outcome:
#  - decks where storage recycling is valid (no DRSDT/DRVDT, no rock
#    compaction): agreement to rounding level;
#  - decks where recycling is disabled (DRSDT/VAPPARS, ROCKCOMP): bitwise
#    agreement, since both modes evaluate the identical storage expression.
# Any larger deviation indicates a genuine bug in the storage-term handling.
#
# Usage:
#   run-storage-cache-equivalence.sh <flow-binary> <deck> <output-dir> [compareECL] [abs-tol] [rel-tol]
#
# If a compareECL binary (from opm-common test_util) is given, summary and
# restart files are compared numerically with the given tolerances
# (defaults 1e-9 / 1e-9); otherwise the unified summary files are compared
# bitwise with cmp and only a warning is printed on mismatch.

set -u

FLOW=${1:?usage: $0 <flow-binary> <deck> <output-dir> [compareECL] [abs-tol] [rel-tol]}
DECK=${2:?missing deck}
OUTDIR=${3:?missing output dir}
COMPARE_ECL=${4:-}
ABSTOL=${5:-1e-9}
RELTOL=${6:-1e-9}

CASENAME=$(basename "${DECK}")
CASENAME=${CASENAME%.DATA}

COMMON_ARGS="--threads-per-process=1 --enable-opm-rst-file=true"

run_one() {
    local cache=$1
    local dir=$2
    mkdir -p "${dir}"
    echo "=== running flow with --enable-storage-cache=${cache}"
    ${FLOW} "${DECK}" --output-dir="${dir}" --enable-storage-cache="${cache}" \
        ${COMMON_ARGS} > "${dir}/flow.log" 2>&1
    local status=$?
    if [ ${status} -ne 0 ]; then
        echo "FAILURE: flow exited with status ${status} (log: ${dir}/flow.log)"
        exit ${status}
    fi
}

run_one true  "${OUTDIR}/cache"
run_one false "${OUTDIR}/nocache"

status=0
if [ -n "${COMPARE_ECL}" ] && [ -x "${COMPARE_ECL}" ]; then
    echo "=== comparing summary with compareECL (abs ${ABSTOL}, rel ${RELTOL})"
    ${COMPARE_ECL} -t SMRY "${OUTDIR}/cache/${CASENAME}" \
        "${OUTDIR}/nocache/${CASENAME}" "${ABSTOL}" "${RELTOL}"
    status=$?
    echo "=== comparing restart with compareECL"
    ${COMPARE_ECL} -t UNRST "${OUTDIR}/cache/${CASENAME}" \
        "${OUTDIR}/nocache/${CASENAME}" "${ABSTOL}" "${RELTOL}" || status=1
else
    echo "=== no compareECL given; bitwise comparison of result files"
    # SMSPEC is skipped: its header embeds the file creation timestamp.
    for f in UNSMRY UNRST; do
        if cmp -s "${OUTDIR}/cache/${CASENAME}.${f}" "${OUTDIR}/nocache/${CASENAME}.${f}"; then
            echo "${f}: bitwise identical"
        else
            echo "${f}: differs (this may be rounding-level; rerun with compareECL for tolerances)"
            status=1
        fi
    done
fi

if [ ${status} -eq 0 ]; then
    echo "T0.7 storage-path equivalence: PASS"
else
    echo "T0.7 storage-path equivalence: DIFFERENCES FOUND (see above)"
fi
exit ${status}
