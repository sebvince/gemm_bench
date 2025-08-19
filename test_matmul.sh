#!/bin/bash

M=$1
N=$2
K=$3

# M=2048
# K=4096
# N=128256
TYPE=f16
python3 ./generate_solution.py ${M}  ${N}  ${K} ${TYPE}

$HOME/iree-build/tools/iree-compile matmul.mlir \
    --iree-hip-target=gfx942 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-codegen-enable-default-tuning-specs=true \
    --iree-codegen-reorder-workgroups-strategy=transpose \
    --iree-opt-level=O3 \
    -o tmp/dispatch.vmfb 

PROFILER="rocprofv3 --att --att-activity 20 --"
PROFILER="rocprofv3 --att --att-activity 10 --att-shader-engine-mask 0x2 --att-simd-select 0xF --att-target-cu 2 --att-buffer-size 1024000000 --"
PROFILER="rocprofv3  --pmc TCC_HIT,TCC_MISS,TCC_HIT_sum,TCC_MISS_sum,TCP_TCC_READ_REQ --output-format csv --stats --output-file res.csv -- "
PROFILER="rocprofv3  --pmc TCC_HIT,TCC_MISS --output-format json --stats --output-file res.json -- "
PROFILER="rocprofv3  --pmc TCC_HIT_RATE_XCC --output-format json --stats --output-file res.json -- "
PROFILER="rocprofv3  --pmc TCC_HIT,TCC_MISS,TCC_EA0_RDREQ --output-format json --stats --output-file res.json -- "

# PROFILER="rocprofv3  --pmc L2CacheHit --output-format csv --stats --output-file res.csv -- "
# PROFILER=

rm -f res.csv_counter_collection.csv

$PROFILER $HOME/iree-build/tools/iree-benchmark-module --benchmark_min_warmup_time=1 --benchmark_repetitions=3 --batch_size=1  --benchmark_min_time=1s \
  --device=hip \
  --device_allocator=caching \
  --module=tmp/dispatch.vmfb \
  --function=matmul \
  --input=${M}x${K}x${TYPE}=@lhs.bin \
  --input=${N}x${K}x${TYPE}=@rhs.bin 


#  python3 rocprof_decode.py res.csv_counter_collection.csv 