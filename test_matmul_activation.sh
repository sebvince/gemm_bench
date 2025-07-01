#!/bin/bash

# prefill_bs4$async_dispatch_21_matmul_like_Dx14336x4096_f8E4M3FNUZxf8E4M3FNUZxf32.kd
# prefill_bs4$async_dispatch_22_matmul_like_Dx14336x4096_f8E4M3FNUZxf8E4M3FNUZxf32.kd
# prefill_bs4$async_dispatch_645_matmul_like_Dx128256x4096_bf16xbf16xf32.kd
# prefill_bs4$async_dispatch_23_matmul_like_Dx4096x14336_f8E4M3FNUZxf8E4M3FNUZxf32.kd

set -eux

ninja -C $HOME/iree-build iree-compile iree-e2e-matmul-test

#llama
# M=8192
# N=14336
# K=4096
# TYPE=f8E4M3FNUZ

# 16384	4096	32768
# 16384	32768	4096
# 16384	4096	8192
# 16384	8704	4096
# 16384	8192	4096

# Shape with discrepancy
# 16384	8192	4096

# Shape lower perf
# 16384	4096	32768

# M=16384
# N=8192
# K=4096
# TYPE=bf16

#Llama.
M=8192
N=14336
K=4096
TYPE=f8E4M3FNUZ
# python3 ./generate_solution.py ${M}  ${N}  ${K} ${TYPE}

# $HOME/iree-build/tools/iree-compile calls.mlir \
#     --iree-hip-target=gfx942 \
#     --iree-hal-target-backends=rocm \
#     -o tmp/calls.vmfb

$HOME/iree-build/tools/iree-compile matmul_activation.mlir \
    --iree-hip-target=gfx942 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-codegen-enable-default-tuning-specs=true \
    --iree-hal-dump-executable-intermediates-to files \
    --iree-hal-dump-executable-files-to files \
    --iree-opt-level=O3 \
    -o tmp/dispatch.vmfb \
    # --mlir-print-ir-after-all \
    # --mlir-disable-threading 2> out.mlir


# $HOME/iree-build/tools/testing/e2e/iree-e2e-matmul-test \
#   --device=hip \
#   --module=tmp/dispatch.vmfb \
#   --module=tmp/calls.vmfb \
#   --acceptable_fp_delta=1e-02
PROFILER="rocprofv3 --att --att-activity 10 --"
# PROFILER="rocprofv3 --att --att-perfcounter-ctrl 3 --att-perfcounters SQ_INSTS_VMEM_RD,SQ_INST_LEVEL_VMEM   --"

# PROFILER=
# items_per_second="$($PROFILER $HOME/iree-build/tools/iree-benchmark-module --benchmark_min_warmup_time=1 --batch_size=1 --benchmark_min_time=1s \
#   --benchmark_format=json \
# --device=hip \
#   --device_allocator=caching \
#   --module=tmp/dispatch.vmfb \
#   --function=matmul \
#   --input=${M}x${K}x${TYPE}=@lhs_mat_act.bin \
#   --input=${N}x${K}x${TYPE}=@rhs_mat_act.bin \
#   --input=${M}x${N}xf16=@out.bin \
#   --input=xf32=1.0 \
#   --input=xf32=1.0 \
#   --input=xf32=1.0 \
#   | grep '"items_per_second":' \
#   | cut -d ':' -f 2)"
# echo "print('%.1f Tflop/s' % (${items_per_second} * ${M} * ${N} * ${K} * 2e-12))" | python

# Reference version
$HOME/iree-build/tools/iree-benchmark-module --benchmark_min_warmup_time=1 --benchmark_repetitions=3 --batch_size=1  --benchmark_min_time=1s \
  --device=hip \
  --device_allocator=caching \
  --module=tmp/dispatch.vmfb \
  --function=matmul \
  --input=${M}x${K}x${TYPE}=@lhs_mat_act.bin \
  --input=${N}x${K}x${TYPE}=@rhs_mat_act.bin \
  --input=${M}x${N}xf16=@out.bin \
  --input=xf32=1.0 \
  --input=xf32=1.0 \
  --input=xf32=1.0
  
# # FP16 Sigmoid version
# $HOME/iree-build/tools/iree-benchmark-module --benchmark_min_warmup_time=1 --benchmark_repetitions=3 --batch_size=1  --benchmark_min_time=1s \
#   --device=hip \
#   --device_allocator=caching \
#   --module=tmp/dispatch.vmfb \
#   --function=matmul \
#   --input=${M}x${K}x${TYPE}=@lhs_mat_act.bin \
#   --input=${N}x${K}x${TYPE}=@rhs_mat_act.bin \
#   --input=xf32=1.0 \
#   --input=xf32=1.0 \
#   --input=xf32=1.0
  