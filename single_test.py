import subprocess
import os
from generate_solution import generate_files
from rocprofResultsParser import parseJsonResults,parseCsvResults

home_dir = os.path.expanduser("~")
IREE_PATH=f'{home_dir}/iree-build/tools'

def compile():
    cmd = [f'{IREE_PATH}/iree-compile',
    'matmul.mlir',
    '--iree-hip-target=gfx942',
    '--iree-hal-target-backends=rocm',
    '--iree-codegen-enable-default-tuning-specs=true',
    '--iree-hip-enable-tensor-ukernels',
    # '--iree-hal-dump-executable-files-to=files',
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

# | 4608 | 16384 | 4096 | -36.6 % | 🔴 8.6 % | 898768.5 | 975723.0 | 634 | NEW: 0.9275725 / OLD :
# | 4864 | 14336 | 4096 | -60.5 % | 🔴 8.3 % | 915093.0 | 990685.5 | 577 | NEW: 0.9511265 / OLD :
# | 4864 | 16384 | 4096 | -59.6 % | 🔴 15.3 % | 909304.0 | 1048391.0 | 623 | NEW : 0.997273 / OLD :
# | 5376 | 14336 | 4096 | -60.2 % | 🔴 14.9 % | 891237.5 | 1023795.0 | 617 | NEW : 0.9694135 / OLD :
# | 5376 | 16384 | 4096 | -56.3 % | 🔴 8.9 % | 1136680.0 | 1237730.5 | 583 | NEW : 1.1952475 / OLD :
# | 7680 | 14336 | 4096 | -39.9 % | 🔴 9.4 % | 1356246.0 | 1483438.5 | 608 | NEW : 1.413891 / OLD :
# | 7936 | 14336 | 4096 | -53.0 % | 🔴 11.6 % | 1342705.0 | 1499120.5 | 622 | NEW : 1.495411 / OLD :
# | 7936 | 16384 | 4096 | -49.3 % | 🔴 12.1 % | 1577752.5 | 1768860.0 | 602 | NEW : 1.7161985 / OLD :
# | 8448 | 14336 | 4096 | -51.5 % | 🔴 10.3 % | 1556081.5 | 1716822.0 | 578 | NEW : 1.679323 / OLD :
# | 8448 | 16384 | 4096 | -48.5 % | 🔴 12.9 % | 1610863.0 | 1819174.0 | 623 | NEW : 1.7714185 / OLD : 1.782656
# | 8704 | 14336 | 4096 | -45.2 % | 🔴 10.4 % | 1566718.0 | 1730161.0 | 591 | NEW : 1.641787 / OLD :1.740531

    M=4864
    N=16384
    K=4096
    tileSize = 256
    dtype='f16'#'f8E4M3FNUZ'
    # dtype='f8E4M3FNUZ'
    isStatic = True
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
