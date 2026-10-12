#!/usr/bin/env bash
# Overnight plane-sextic run: phase A (random scan with large diverse pools), three births segments (B1 default,
# B2 nest-biased, B3 frontier-biased), phase C (volumes, certification of all types, export, verification, plots).
# Resumable: finished phases are skipped (their output files exist); phase A resumes from its checkpoint.
#
#   setsid nohup scripts/sextic_overnight.sh > pools/sextic_overnight.log 2>&1 &
#   MIN_A=3 MIN_B=2 OUT=pools/overnight_test REPS=1 scripts/sextic_overnight.sh     # smoke test
#   KIND=hessian setsid nohup scripts/sextic_overnight.sh > pools/hessian_overnight.log 2>&1 &   # Hessians of quintics
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
KIND=${KIND:-plain}
if [ "$KIND" = hessian ]; then
    DEFAULT_OUT=pools/hessian_overnight; DATA_DIR=data/hessian_types
    PREV=(data/strategies/*.txt data/cache_H_deg5_*.txt data/symmetric/*.txt data/degenerate/*.txt)
else
    DEFAULT_OUT=pools/sextic_overnight; DATA_DIR=data/sextic_types
    PREV=(pools/sextic_births/archive.npz pools/sextic_plain/phase_a.npz pools/sextic_overnight/B3/archive.npz)
fi
OUT=${OUT:-$DEFAULT_OUT}
MIN_A=${MIN_A:-90}
MIN_B=${MIN_B:-90}
DIVERSE=${DIVERSE:-20000}
REPS=${REPS:-3}
REPS_HIGH=${REPS_HIGH:-5}
DATA_OUT=${DATA_OUT:-$DATA_DIR/overnight.json}
mkdir -p "$OUT"
status() { echo "$(date '+%F %T') $*" | tee -a "$OUT/status.txt"; }

if [ ! -f "$OUT/A/diverse.npz" ]; then
    status "phase A: $MIN_A min"
    $PY scripts/sextic_types.py --kind "$KIND" --stage a --minutes-a "$MIN_A" --checkpoint 10 --pool 2000 \
        --diverse "$DIVERSE" \
        --out "$OUT/A" > "$OUT/A.log" 2>&1
fi
status "phase A done: $(tail -n 1 "$OUT/A.log")"

seeds=("$OUT/A/phase_a.npz" "$OUT/A/diverse.npz")
for p in "${PREV[@]}"; do [ -f "$p" ] && seeds+=("$p"); done
prev=""
for seg in B1:default B2:nest B3:frontier; do
    name=${seg%%:*}
    mode=${seg##*:}
    if [ ! -f "$OUT/$name/phase_b.npz" ]; then
        status "segment $name ($mode): $MIN_B min"
        args=("${seeds[@]}")
        [ -n "$prev" ] && args=("$prev/archive.npz" "$OUT/A/diverse.npz")
        $PY scripts/sextic_births.py --kind "$KIND" --minutes "$MIN_B" --mode "$mode" --checkpoint 10 --no-certify \
            --pools "${args[@]}" --out "$OUT/$name" > "$OUT/$name.log" 2>&1
    fi
    status "segment $name done: $(grep '^done' "$OUT/$name.log" | tail -n 1)"
    prev="$OUT/$name"
done

if [ ! -f "$OUT/A/summary.json" ]; then
    status "phase C: finish (volumes, certification)"
    $PY scripts/sextic_types.py --kind "$KIND" --stage finish --out "$OUT/A" --phase-b "$OUT/B3/phase_b.npz" \
        --reps "$REPS" \
        --reps-high "$REPS_HIGH" --data-out "$DATA_OUT" > "$OUT/C.log" 2>&1
fi
if [ "$OUT" = "$DEFAULT_OUT" ]; then
    $PY scripts/export_type_certificates.py >> "$OUT/C.log" 2>&1
    $PY verify/verify_certificate.py -q "$DATA_DIR"/certificates/overnight_*.json > "$OUT/verify.log" 2>&1 || true
    $PY scripts/plot_sextic_types.py --kind "$KIND" >> "$OUT/C.log" 2>&1
    status "verified: $(grep -c '=> VERIFIED' "$OUT/verify.log") / $(ls "$DATA_DIR"/certificates/overnight_*.json | wc -l)"
fi
status "all done"
