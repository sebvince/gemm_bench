#!/bin/bash

# prefill_bs4$async_dispatch_21_matmul_like_Dx14336x4096_f8E4M3FNUZxf8E4M3FNUZxf32.kd
# prefill_bs4$async_dispatch_22_matmul_like_Dx14336x4096_f8E4M3FNUZxf8E4M3FNUZxf32.kd


# prefill_bs4$async_dispatch_645_matmul_like_Dx128256x4096_bf16xbf16xf32.kd

# prefill_bs4$async_dispatch_23_matmul_like_Dx4096x14336_f8E4M3FNUZxf8E4M3FNUZxf32.kd

set -eux

ninja -C $HOME/iree-build iree-compile iree-e2e-matmul-test

M=16384
N=32768
K=4096

python3 ./generate_solution.py ${M}  ${N}  ${K}

$HOME/iree-build/tools/iree-compile calls.mlir \
    --iree-hip-target=gfx942 \
    --iree-hal-target-backends=rocm \
    -o tmp/calls.vmfb

$HOME/iree-build/tools/iree-compile matmul.mlir \
    --iree-hip-target=gfx942 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-codegen-enable-default-tuning-specs=true \
    --iree-hal-dump-executable-binaries-to .  \
    --iree-hal-dump-executable-sources-to . \
    --iree-hal-dump-executable-intermediates-to . \
    --iree-hal-dump-executable-files-to . \
    --iree-opt-level=O3 \
    -o tmp/dispatch.vmfb \
    # --mlir-print-ir-after-all \
    # --mlir-disable-threading 2> out.mlir

# $HOME/iree-build/tools/iree-compile matmul14336.mlir \
#     --iree-hip-target=gfx942 \
#     --iree-hal-target-backends=rocm \
#     --mlir-disable-threading \
#     --iree-codegen-enable-default-tuning-specs=true \
#     --iree-hal-dump-executable-binaries-to .  \
#     --iree-hal-dump-executable-sources-to . \
#     --iree-hal-dump-executable-intermediates-to . \
#     --iree-hal-dump-executable-files-to . \
#     -o tmp/dispatch.vmfb


$HOME/iree-build/tools/testing/e2e/iree-e2e-matmul-test \
  --device=hip \
  --module=tmp/dispatch.vmfb \
  --module=tmp/calls.vmfb \
  --acceptable_fp_delta=1e-02
PROFILER="rocprofv3 --att --"
PROFILER=
# items_per_second="$($PROFILER $HOME/iree-build/tools/iree-benchmark-module --benchmark_min_warmup_time=1 --batch_size=1 --benchmark_min_time=1s \
#   --benchmark_format=json \
#   --device=hip \
#   --device_allocator=caching \
#   --module=tmp/dispatch.vmfb \
#   --function=matmul \
#   --input=${M}x${K}xf8E4M3FNUZ=@lhs.bin \
#   --input=${N}x${K}xf8E4M3FNUZ=@rhs.bin \
#   | grep '"items_per_second":' \
#   | cut -d ':' -f 2)"

# echo "print('%.1f Tflop/s' % (${items_per_second} * ${M} * ${N} * ${K} * 2e-12))" | python


$HOME/iree-build/tools/iree-benchmark-module --benchmark_min_warmup_time=1 --benchmark_repetitions=3 --batch_size=1  --benchmark_min_time=1s \
  --device=hip \
  --device_allocator=caching \
  --module=tmp/dispatch.vmfb \
  --function=matmul \
  --input=${M}x${K}xf8E4M3FNUZ=@lhs.bin \
  --input=${N}x${K}xf8E4M3FNUZ=@rhs.bin 