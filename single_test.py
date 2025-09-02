import subprocess
import os
from generate_solution import generate_files
from rocprofResultsParser import parseJsonResults

home_dir = os.path.expanduser("~")
IREE_PATH=f'{home_dir}/iree-build/tools'

def compile(useTranspose = False):
    transposeStrategy = 'transpose' if useTranspose else 'none'
    cmd = [f'{IREE_PATH}/iree-compile',
    'matmul.mlir',
    '--iree-hip-target=gfx942',
    '--iree-hal-target-backends=rocm',
    '--iree-codegen-enable-default-tuning-specs=true',
    f'--iree-codegen-reorder-workgroups-strategy={transposeStrategy}',
    '--iree-opt-level=O3',
    '-o','tmp/dispatch.vmfb']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print("STDERR:", result.stderr)

def run(M,N,K,TYPE, profilePerXCD = True):

    counters = ['TCC_HIT','TCC_MISS','TCC_EA0_RDREQ']
    output_format = 'json' if profilePerXCD else 'csv'
    cmd = [ 'rocprofv3',
            '--pmc', ",".join(counters), 
            '--output-format', output_format,
            '--output-file', f'res.{output_format}',
            '--',
            f'{IREE_PATH}/iree-benchmark-module', 
            '--benchmark_min_warmup_time=0.1',
            '--benchmark_repetitions=2',
            '--batch_size=1',
            '--benchmark_min_time=0.1s',
            '--device=hip',
            '--device_allocator=caching',
            '--module=tmp/dispatch.vmfb',
            '--function=matmul',
            f'--input={M}x{K}x{TYPE}=@lhs.bin',
            f'--input={N}x{K}x{TYPE}=@rhs.bin']

    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout:
        print(line, end='')  # Output each line as it arrives
   
    if result.stderr:
        print("STDERR:", result.stderr)
   
if __name__ == "__main__":  
    M=8192
    N=128256
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
    parseJsonResults(filename)
