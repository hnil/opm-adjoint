#!/bin/bash
# Finite-difference verification of the adjoint end-point scaling
# gradients (test level T5), specialized for adjoint_tests/MODEL_1D_DEBUG.
#
# The stock deck has no ENDSCALE; this script derives an end-point
# scaling variant: ENDSCALE in RUNSPEC plus a per-cell include for the
# tested keyword in PROPS, initialized to the table value (SWL -> table
# connate 0.0, KRW -> table max 1.0, ...) so the baseline run is
# identical to the unscaled deck. The deck initializes with explicit
# PRESSURE/SWAT/SGAS (not EQUIL), so the initial state is independent of
# the end points and the trajectory-only adjoint gradient is the full
# gradient.
#
# For each cell i:
#   FD:      dJ/dtheta_i ~ (J(theta_i + d) - J(theta_i - d)) / (2 d)
#   adjoint: ADJOINT_GRADIENTS_ENDPOINT_<KW>.txt from --adjoint-endpoints
#
# Usage:
#   run-adjoint-fd-endpoint-test.sh <flow_adjoint> <inputfiles-dir> <output-dir> \
#       [keyword] [base-value] [delta] [rel-tol] [objective] [orat]

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <inputfiles-dir> <output-dir> [keyword] [base] [delta] [rel-tol] [objective] [orat]}
INPUTDIR=${2:?missing inputfiles dir}
OUTDIR=${3:?missing output dir}
KEYWORD=${4:-SWL}
BASEVAL=${5:-0.0}
DELTA=${6:-1e-5}
RELTOL=${7:-1e-3}
OBJECTIVE=${8:-pressure-average}
ORAT=${9:-}

DECKNAME=MODEL_1D_DEBUG
NCELLS=3

# --check-satfunc-consistency=false: a perturbed SWL formally violates
# SWL <= SWCR (the table has SWCR = SWL = 0); harmless for FD purposes.
STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-7 \
 --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9 \
 --check-satfunc-consistency=false \
 --adjoint-objective=${OBJECTIVE}"

# Copy the inputfiles and turn the deck into an end-point scaling deck.
make_input() { # <dstdir> <cell-idx-or-(-1)> <delta>
    local dst=$1 idx=$2 delta=$3
    rm -rf "${dst}"
    cp -r "${INPUTDIR}" "${dst}"
    python3 - "${dst}" "${DECKNAME}" "${KEYWORD}" "${BASEVAL}" "${NCELLS}" "${idx}" "${delta}" << 'EOF'
import sys
dst, deck, kw, base, n, idx, delta = sys.argv[1:8]
base, n, idx, delta = float(base), int(n), int(idx), float(delta)
vals = [base + (delta if i == idx else 0.0) for i in range(n)]
with open(f"{dst}/endpoint.txt", "w") as f:
    f.write(kw + "\n")
    for v in vals:
        f.write(f"{v:.16e}\n")
    f.write("/\n")
path = f"{dst}/{deck}.DATA"
text = open(path).read()
text = text.replace("START ", "ENDSCALE\n/\n\nSTART ", 1)
text = text.replace("INCLUDE\nswof.txt\n/",
                    "INCLUDE\nswof.txt\n/\n\nINCLUDE\nendpoint.txt\n/", 1)
open(path, "w").write(text)
EOF
    if [ -n "${ORAT}" ]; then
        local wcf wellname
        for wcf in "${dst}"/wconprodstep_*.txt; do
            wellname=$(awk 'NR==2 {print $1}' "${wcf}")
            printf 'WCONPROD\n%s OPEN ORAT %s 4* 50.0 /\n/\n' "${wellname}" "${ORAT}" > "${wcf}"
        done
    fi
}

run_J() { # <inputdir> <rundir>  -> prints J
    local inputdir=$1 rundir=$2
    mkdir -p "${rundir}"
    ${FLOW} "${inputdir}/${DECKNAME}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-save=true > "${rundir}/fwd.log" 2>&1 || { echo FAIL; return; }
    ${FLOW} "${inputdir}/${DECKNAME}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-mode=objective > "${rundir}/obj.log" 2>&1 || { echo FAIL; return; }
    grep -o "Adjoint objective J = [0-9eE.+-]*" "${rundir}/obj.log" | awk '{print $5}'
}

mkdir -p "${OUTDIR}"

echo "=== base run (${KEYWORD} = ${BASEVAL}): forward + adjoint gradient"
BASE=${OUTDIR}/base
mkdir -p "${BASE}"
make_input "${OUTDIR}/inputfiles_base" -1 0
${FLOW} "${OUTDIR}/inputfiles_base/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-save=true > "${BASE}/fwd.log" 2>&1 || { echo "base forward failed"; exit 1; }
${FLOW} "${OUTDIR}/inputfiles_base/${DECKNAME}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-mode=gradient --adjoint-endpoints="${KEYWORD}" \
    > "${BASE}/grad.log" 2>&1 || { echo "gradient run failed"; exit 1; }
GRADFILE=${BASE}/${DECKNAME}.ADJOINT_GRADIENTS_ENDPOINT_${KEYWORD}.txt
[ -f "${GRADFILE}" ] || { echo "no gradient file ${GRADFILE}"; exit 1; }

echo "=== ${NCELLS} cells, FD delta ${DELTA}"

status=0
i=0
while [ $i -lt ${NCELLS} ]; do
    for sign in 1 -1; do
        dir=${OUTDIR}/fd_${i}_${sign}
        make_input "${OUTDIR}/inputfiles_${i}_${sign}" $i $(python3 -c "print(${sign} * ${DELTA})")
        J=$(run_J "${OUTDIR}/inputfiles_${i}_${sign}" "${dir}")
        [ "$J" = "FAIL" ] && { echo "FD run cell $i sign $sign failed"; exit 1; }
        if [ "$sign" = "1" ]; then JP=$J; else JM=$J; fi
    done
    read FD ADJ ERR MAXOK << EOF
$(python3 -c "
fd = ($JP - $JM) / (2 * $DELTA)
adj = float(open('$GRADFILE').read().splitlines()[$((i + 1))])
err = abs(fd - adj) / max(abs(fd), abs(adj), 1e-300)
print(fd, adj, err, 1 if err < $RELTOL else 0)")
EOF
    printf "cell %2d: FD %15.6e  adjoint %15.6e  rel.err %9.2e\n" $i $FD $ADJ $ERR
    [ "$MAXOK" = "1" ] || status=1
    i=$((i + 1))
done

if [ ${status} -eq 0 ]; then
    echo "T5 adjoint-vs-FD (endpoint ${KEYWORD}): PASS (rel tol ${RELTOL})"
else
    echo "T5 adjoint-vs-FD (endpoint ${KEYWORD}): FAIL"
fi
exit ${status}
