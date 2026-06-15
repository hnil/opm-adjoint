#!/bin/bash
# Finite-difference verification of the adjoint pore-volume / porosity
# gradient on a GROUP-CONTROLLED deck (SPE1_GRPCTRL: SPE1 with the
# producer under GCONPROD group control). This is the field-relevant
# check: it proves the adjoint gradient is correct when wells follow a
# group target (guide-rate allocation), not just for single-well control.
#
# The deck has per-cell PORO (300 explicit values), so individual cells
# can be perturbed. For each sampled cell i:
#   FD:      dJ/dporo_i ~ (J(poro_i + d) - J(poro_i - d)) / (2 d)
#   adjoint: dJ/dporo_i = g_pvmult_i / poro_i
# with J = average reservoir pressure at the final step.
#
# Usage:
#   run-adjoint-fd-grpctrl-test.sh <flow_adjoint> <SPE1_GRPCTRL.DATA> <out-dir> [delta] [rel-tol]

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <deck.DATA> <out-dir> [delta] [rel-tol]}
DECK=${2:?missing deck}
OUTDIR=${3:?missing output dir}
DELTA=${4:-1e-3}
RELTOL=${5:-5e-2}
PORO0=0.3

# Tight tolerances so the FD signal (small for a constant-rate group
# target) sits above the solver convergence noise; fixed report-step
# stepping so a perturbation cannot change the substep sequence.
STRICT="--enable-storage-cache=false --enable-adaptive-time-stepping=false \
 --threads-per-process=1 --tolerance-cnv=1e-9 --tolerance-mb=1e-11 \
 --adjoint-objective=pressure-average"

mkdir -p "${OUTDIR}"
cp "${DECK}" "${OUTDIR}/base.DATA"
# the deck is base.DATA, so the case name (and gradient file prefix) is BASE
CASE=BASE

# base adjoint gradient
${FLOW} "${OUTDIR}/base.DATA" --output-dir="${OUTDIR}" ${STRICT} \
    --adjoint-file="${OUTDIR}/a" --adjoint-save=true > "${OUTDIR}/fwd.log" 2>&1 \
    || { echo "base forward failed"; tail -3 "${OUTDIR}/fwd.log"; exit 1; }
${FLOW} "${OUTDIR}/base.DATA" --output-dir="${OUTDIR}" ${STRICT} \
    --adjoint-file="${OUTDIR}/a" --adjoint-mode=gradient > "${OUTDIR}/grad.log" 2>&1 \
    || { echo "base gradient failed"; tail -3 "${OUTDIR}/grad.log"; exit 1; }
GRAD="${OUTDIR}/${CASE}.ADJOINT_GRADIENTS_PV.txt"
[ -f "${GRAD}" ] || { echo "no gradient file ${GRAD}"; exit 1; }

perturb() { # <file> <cell-idx> <delta>
    python3 - "$1" "$2" "$3" << 'EOF'
import sys
f, idx, delta = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
lines = open(f).read().split("\n"); out = []; inpor = False; cnt = 0
for l in lines:
    t = l.strip()
    if t == 'PORO':
        inpor = True; out.append(l); continue
    if inpor and t == '0.3':
        if cnt == idx:
            l = f"{0.3 + delta:.10f}"
        cnt += 1
        out.append(l); continue
    if inpor and t == '/':
        inpor = False
    out.append(l)
open(f, 'w').writelines("\n".join(out))
EOF
}

getJ() { # <dir>
    ${FLOW} "$1/base.DATA" --output-dir="$1" ${STRICT} --adjoint-file="$1/a" \
        --adjoint-save=true > "$1/f.log" 2>&1
    ${FLOW} "$1/base.DATA" --output-dir="$1" ${STRICT} --adjoint-file="$1/a" \
        --adjoint-mode=objective > "$1/o.log" 2>&1
    grep -o "Adjoint objective J = [0-9eE.+-]*" "$1/o.log" | awk '{print $5}'
}

echo "=== group-control FD vs adjoint (delta ${DELTA}, rel tol ${RELTOL}) ==="
echo "cell        FD          adjoint(g/poro)   rel.err"
status=0
for c in 0 50 100 150 220; do
    for sgn in + -; do
        d="${OUTDIR}/p_${c}_${sgn}"; mkdir -p "$d"; cp "${OUTDIR}/base.DATA" "$d/"
        perturb "$d/base.DATA" "$c" "${sgn}${DELTA}"
        J=$(getJ "$d")
        [ "$sgn" = "+" ] && JP=$J || JM=$J
    done
    python3 -c "
fd = (${JP} - (${JM})) / (2*${DELTA})
g = float(open('${GRAD}').read().splitlines()[$((c+1))]) / ${PORO0}
err = abs(fd-g)/max(abs(fd), abs(g), 1e-30)
print('%4d  %14.6e  %14.6e  %9.2e  %s' % (${c}, fd, g, err, 'OK' if err < ${RELTOL} else 'FAIL'))
import sys; sys.exit(0 if err < ${RELTOL} else 1)" || status=1
done

if [ ${status} -eq 0 ]; then
    echo "T5 adjoint-vs-FD (group control, porosity): PASS (rel tol ${RELTOL})"
else
    echo "T5 adjoint-vs-FD (group control, porosity): FAIL"
fi
exit ${status}
