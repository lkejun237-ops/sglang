#!/usr/bin/env bash
# Lynn SP tile-parallel acceptance (PR #298, design §5.1).
# SP1, then every degree in SP_DEGREES, then report. Exit status is the max rc.
# REFERENCE_OUTPUT, when set, is a directory of precomputed reference/R frames.
set -uo pipefail

: "${OUT:?}" "${INPUTS:?}" "${CHECKPOINT:?}" "${CHECKPOINT_SHA256:?}" "${MODEL_DIR:?}"
: "${SGLANG_DIR:?}" "${OVERLAY_041:?}"
PYTHON=${PYTHON:-python}
SP_DEGREES=${SP_DEGREES:-2}
BENCH="$SGLANG_DIR/benchmark/zing_h3_causal"
mkdir -p "$OUT"
common=(--inputs "$INPUTS" --checkpoint "$CHECKPOINT" --model-dir "$MODEL_DIR")
ref_args=()
if [ -n "${REFERENCE_OUTPUT:-}" ]; then
  ref_args=(--reference-output "$REFERENCE_OUTPUT")
fi

step() {  # name, timeout, PYTHONPATH, command...
  local name=$1 limit=$2 path=$3; shift 3
  echo "=== $name $(date -u +%FT%TZ)"
  PYTHONPATH="$path" timeout "$limit" "$@" > "$OUT/log-$name.txt" 2>&1
  local rc=$?
  echo "=== $name rc=$rc"; tail -n 20 "$OUT/log-$name.txt"
  return $rc
}

# Freeze intent before the first worker; old metrics cannot satisfy this run.
manifest_args=()
if [ -n "${REFERENCE_OUTPUT:-}" ]; then manifest_args=(--reference); fi
"$PYTHON" "$BENCH/lynn_acceptance_manifest.py" --output "$OUT" --inputs "$INPUTS" \
  --sp-degrees 1 $SP_DEGREES --lifecycle-sample video-480p-checkpoint_generated-vbh000169_h3 \
  "${manifest_args[@]}" || exit $?

rcs=()
for degree in 1 $SP_DEGREES; do
  step "sp$degree" "${T_SP:-30m}" "$OVERLAY_041:$SGLANG_DIR/python" \
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node "$degree" \
    "$BENCH/validate_lynn_sp_tiles.py" decode --output "$OUT" "${common[@]}" \
    --checkpoint-sha256 "$CHECKPOINT_SHA256" ${ref_args[@]+"${ref_args[@]}"}
  rcs+=($?)
done
step report 10m "$OVERLAY_041:$SGLANG_DIR/python" "$PYTHON" "$BENCH/validate_lynn_sp_tiles.py" report --output "$OUT"
rcs+=($?)
cat "$OUT/log-report.txt"

rc=0
for value in "${rcs[@]}"; do
  if [ "$value" -gt "$rc" ]; then rc=$value; fi
done
exit "$rc"
