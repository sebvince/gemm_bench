#!/bin/bash

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

#M=8192
M=2048
K=8192
N=128256
TYPE=f16



python3 ./generate_solution.py ${M}  ${N}  ${K} ${TYPE}

# $HOME/iree-build/tools/iree-compile calls.mlir \
#     --iree-hip-target=gfx942 \
#     --iree-hal-target-backends=rocm \
#     -o tmp/calls.vmfb

$HOME/iree-build/tools/iree-compile matmul.mlir \
    --iree-hip-target=gfx942 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-codegen-enable-default-tuning-specs=true \
    --iree-hal-dump-executable-intermediates-to files \
    --iree-hal-dump-executable-files-to files \
    --iree-opt-level=O3 \
    -o tmp/dispatch.vmfb 
    # --mlir-print-ir-after-all \
    # --mlir-disable-threading 2> out.mlir
# --iree-codegen-reorder-workgroups-strategy=transpose \
# --mlir-print-ir-module-scope\

# $HOME/iree-build/tools/testing/e2e/iree-e2e-matmul-test \
#   --device=hip \
#   --module=tmp/dispatch.vmfb \
#   --module=tmp/calls.vmfb \
#   --acceptable_fp_delta=1e-02
PROFILER="rocprofv3 --att --att-activity 20 --"
# PROFILER="rocprofv3 --att --att-perfcounter-ctrl 3 --att-perfcounters SQ_INSTS_VMEM_RD,SQ_INST_LEVEL_VMEM   --"
PROFILER="rocprofv3 --att --att-activity 10 --att-simd-select 0xF --att-buffer-size 1024000000 --"
PROFILER="rocprofv3  --pmc TCC_HIT,TCC_MISS,TCP_TCC_READ_REQ --output-format csv --stats --output-file res.csv -- "
# PROFILER=

# PROFILER=
# items_per_second="$($PROFILER $HOME/iree-build/tools/iree-benchmark-module --benchmark_min_warmup_time=1 --batch_size=1 --benchmark_min_time=1s \
#   --benchmark_format=json \
#   --device=hip \
#   --device_allocator=caching \
#   --module=tmp/dispatch.vmfb \
#   --function=matmul \
#   --input=${M}x${K}x${TYPE}=@lhs.bin \
#   --input=${N}x${K}x${TYPE}=@rhs.bin \
#   | grep '"items_per_second":' \
#   | cut -d ':' -f 2)"

# echo "print('%.1f Tflop/s' % (${items_per_second} * ${M} * ${N} * ${K} * 2e-12))" | python

rm -f res.csv_counter_collection.csv

$PROFILER $HOME/iree-build/tools/iree-benchmark-module --benchmark_min_warmup_time=1 --benchmark_repetitions=3 --batch_size=1  --benchmark_min_time=1s \
  --device=hip \
  --device_allocator=caching \
  --module=tmp/dispatch.vmfb \
  --function=matmul \
  --input=${M}x${K}x${TYPE}=@lhs.bin \
  --input=${N}x${K}x${TYPE}=@rhs.bin 


 python3 rocprof_decode.py res.csv_counter_collection.csv 