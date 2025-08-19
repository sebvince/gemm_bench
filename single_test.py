import subprocess
import os
from generate_solution import generate_files
from parse_json import parseJsonResults

home_dir = os.path.expanduser("~")
IREE_PATH=f'{home_dir}/iree-build/tools'

def compile():
    cmd = [f'{IREE_PATH}/iree-compile',
    'matmul.mlir',
    '--iree-hip-target=gfx942',
    '--iree-hal-target-backends=rocm',
    '--iree-codegen-enable-default-tuning-specs=true',
    '--iree-codegen-reorder-workgroups-strategy=transpose',
    '--iree-opt-level=O3',
    '-o','tmp/dispatch.vmfb']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print("STDERR:", result.stderr)

def run(M,N,K,TYPE,useProfiler = True):
    cmd = [ 'rocprofv3',
            '--pmc','TCC_HIT,TCC_MISS', 
            # '--input /home/svince/therock942/share/rocprofiler-sdk/counter_defs.yaml',
            # ,TCC_EA0_RDREQ_DRAM_sum',
            '--output-format json',
            '--output-file res.json',
            '--',
            f'{IREE_PATH}/iree-benchmark-module', 
            '--benchmark_min_warmup_time=1',
            '--benchmark_repetitions=3',
            '--batch_size=1',
            '--benchmark_min_time=1s',
            '--device=hip',
            '--device_allocator=caching',
            '--module=tmp/dispatch.vmfb',
            '--function=matmul',
            f'--input={M}x{K}x{TYPE}=@lhs.bin',
            f'--input={N}x{K}x{TYPE}=@rhs.bin']

    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    for line in result.stdout:
        print(line, end='')  # Output each line as it arrives
    result.wait()
    if result.stderr:
        print("STOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("CMD:"," ".join(cmd))

if __name__ == "__main__":  
    M=4096
    N=4096
    K=4096
    dtype='f16'
    filename ='res.json_results.json'
   
    if os.path.exists(filename):
        os.remove(filename)
        print("Removed result file.")

    generate_files(M,N,K,dtype)
    print('Compiling...')
    compile()
    print('Running...')
    run(M,N,K,dtype)
    parseJsonResults(filename,M,N,K)