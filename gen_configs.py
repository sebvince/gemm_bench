import csv
import subprocess
from rocprof_decode import parseRocprofResults

# M = [1024,2048,4096,8192,16384,32768,65536,128256,131072]
# N = [1024,2048,4096,8192,16384,32768,65536,128256,131072]
# M = [1024,2048,4096,8192,16384]
M = [1024, 2048,4096,8192,16384]
N = [32768,65536,128256]
K = [2048, 4096]

test_cases = []

for m in M:
    for n in N:
        for k in K:
            size_gb = ((m*k + n*k + n*m)*2)/1e9
            if size_gb<6:
                test_cases.append((m,n,k))
                # test_cases.append((n,m,k))

data = []
nb_tests = len(test_cases)
for index in range(nb_tests):
    (m,n,k) = test_cases[index]
    print(f'Matmul_{m}_{n}_{k} : {index}')
    result = subprocess.run(['./test_matmul.sh', f'{m}', f'{n}', f'{k}'], capture_output=True, text=True)
    # print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    (median_TCC_HIT_RATE,tflops,shape) = parseRocprofResults("res.csv_counter_collection.csv")
    data.append([shape[0],shape[1],shape[2], median_TCC_HIT_RATE, tflops])
    print("RES ",data[len(data)-1])
    # Write to CSV file
    with open('summary.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['M', 'N', 'K', 'Median_TCC_HIT_RATE', 'TFLOPS'])
        writer.writerows(data)

# for value in data:
#     print(value)


