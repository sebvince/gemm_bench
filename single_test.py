import subprocess
import os
from generate_solution import generate_files
from rocprofResultsParser import parseJsonResults,parseCsvResults

home_dir = os.path.expanduser("~")
IREE_PATH=f'{home_dir}/iree-build/tools'

def compile(useTranspose = False):
    transposeStrategy = 'transpose' if useTranspose else 'none'
    cmd = [f'{IREE_PATH}/iree-compile',
    'matmul.mlir',
    '--iree-hip-target=gfx942',
    '--iree-hal-target-backends=rocm',
    '--iree-codegen-enable-default-tuning-specs=true',
    '--iree-hip-enable-tensor-ukernels',
    '--iree-hal-dump-executable-files-to=files',
    f'--iree-codegen-reorder-workgroups-strategy={transposeStrategy}',
    '--iree-opt-level=O3',
    '-o','tmp/dispatch.vmfb']
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print("STDERR:", result.stderr)

def run(M,N,K,TYPE, profilePerXCD = False):
    counters = ['TCC_HIT','TCC_MISS','TCC_EA0_RDREQ','TCC_EA0_RDREQ_LEVEL']
    output_format = 'json' if profilePerXCD else 'csv'
    cmd = [ 'rocprofv3',
            '--pmc', ",".join(counters), 
            '--output-format', output_format,
            '--output-file', f'res.{output_format}',
            '--',
            f'{IREE_PATH}/iree-benchmark-module', 
            '--benchmark_min_warmup_time=0.1',
            '--benchmark_repetitions=5',
            '--batch_size=1',
            '--benchmark_min_time=0.1s',
            '--device=hip',
            '--device_allocator=caching',
            '--module=tmp/dispatch.vmfb',
            '--function=matmul',
            f'--input={M}x{K}x{TYPE}=@lhs.bin',
            f'--input={N}x{K}x{TYPE}=@rhs.bin']

    while True:
        print(" ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=100)
            for line in result.stdout:
                print(line, end='')  # Output each line as it arrives
            break  # Exit loop if successful
        except subprocess.TimeoutExpired:
            print(f"Command timed out after 100 seconds. Retrying...")   
   
if __name__ == "__main__":  


    M=8192
    N=128256
    K=4096
    tileSize = 128
    dtype='f16'#'f8E4M3FNUZ'
    dtype='f8E4M3FNUZ'
    isStatic = False
    profilePerXCD = False
    transposedReorder = False
    filename ='res.csv_counter_collection.csv'

    if profilePerXCD:
        filename ='res.json_results.json'
   
    if os.path.exists(filename):
        os.remove(filename)
        print("Removed result file.")

    generate_files(M,N,K,dtype,isStatic,tileSize)
    print('Compiling...')
    compile(transposedReorder)
    print('Running...')
    run(M,N,K,dtype,profilePerXCD)
    if profilePerXCD:
        parseJsonResults(filename)
    else:
        (median_TCC_HIT_RATE,median_time_ns,median_TCC_EA0_RDREQ , median_EA0_RDREQ_LEVEL) = parseCsvResults(filename)
        print("TCC_HIT_RATE:", median_TCC_HIT_RATE)
        print("EA0_RDREQ_LEVEL:", median_EA0_RDREQ_LEVEL)
        print("median_TCC_EA0_RDREQ:", median_TCC_EA0_RDREQ)
        print("EA Latency:", median_EA0_RDREQ_LEVEL/median_TCC_EA0_RDREQ)
        print("Time (ms):", median_time_ns/1e6)
        print("FLOPS (ms):", M*N*K*2/median_time_ns/1e3)
