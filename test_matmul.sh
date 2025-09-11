#!/bin/bash

M=$1
N=$2
K=$3

TYPE=f16
python3 ./generate_solution.py ${M} ${N} ${K} ${TYPE}

$HOME/iree-build/tools/iree-compile matmul.mlir \
    --iree-hip-target=gfx942 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-codegen-enable-default-tuning-specs=true \
    --iree-codegen-reorder-workgroups-strategy=none \
    --iree-hal-dump-executable-benchmarks-to=files \
    --iree-config-add-tuner-attributes \
    --iree-opt-level=O3 \
    -o tmp/dispatch.vmfb 

$HOME/iree-build/tools/iree-compile calls.mlir \
    --iree-hip-target=gfx942 \
    --iree-hal-target-backends=rocm \
    -o tmp/calls.vmfb

$HOME/iree-build/tools/testing/e2e/iree-e2e-matmul-test \
  --device=hip \
  --module=tmp/dispatch.vmfb \
  --module=tmp/calls.vmfb \
  --acceptable_fp_delta=1e-02

$PROFILER $HOME/iree-build/tools/iree-benchmark-module --benchmark_min_warmup_time=1 --benchmark_repetitions=3 --batch_size=1  --benchmark_min_time=1s \
  --device=hip \
  --device_allocator=caching \
  --module=tmp/dispatch.vmfb \
  --function=matmul \
  --input=${M}x${K}x${TYPE}=@lhs.bin \
  --input=${N}x${K}x${TYPE}=@rhs.bin 
