import csv
import os
import sys
from generate_solution import generate_files
from rocprofResultsParser import parseJsonResults,parseCsvResults
from single_test import compile,run

# M = [2048,4096,4352,8192,8448,16384,16640,32768,33024,65536,65792,128256,131072]
# N = [2048,4096,4352,8192,8448,16384,16640,32768,33024,65536,65792,128256,131072]
M=[2048,4096,4608,4864,5376,7680,7936,8192,8448,8704,14080,14336,16128,16384,16640,32512,32768,33024,65536,65792,66048,128256,130816,131072]
N=[2048,4096,4608,4864,5376,7680,7936,8192,8448,8704,14080,14336,16128,16384,16640,32512,32768,33024,65536,65792,66048,128256,130816,131072]

K = [4096]

test_cases = []

for m in M:
    for n in N:
        for k in K:
            size_gb = ((m*k + n*k + n*m)*2)/1e9
            if size_gb<6:
                test_cases.append((m,n,k))


def getLastId(fileName):
    last_id = -1
    if os.path.exists(fileName):
        with open(fileName, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                if row:  
                    last_id = int(row[0]) 
    else:
        with open(fileName, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID','M', 'N', 'K', 'TCC_HIT_RATE', 'time_ns','TCC_EA0_RDREQ','EA0_RDREQ_LEVEL'])
    return last_id

fileName = 'summary'

# Options
# dtype='f16'
isStatic = sys.argv[1]=='1'
transposedReorder = sys.argv[2]=='1'
dtype = sys.argv[3]

print("isStatic:", isStatic)
print("transposedReorder:", transposedReorder)

fileName = fileName + '_static' if isStatic else fileName + '_dynamic'
fileName = fileName + '_transposed' if transposedReorder else fileName + '_regular'
fileName += f'_{dtype}'
fileName += '.csv'
startId = getLastId(fileName)+1
print("Output to " + fileName)

data = []
nb_tests = len(test_cases)


for index in range(startId,nb_tests):
    (m,n,k) = test_cases[index]
    print(f'Matmul_{m}_{n}_{k} : {index}')
    generate_files(m,n,k,dtype,isStatic)
    compile(transposedReorder)
    run(m,n,k,dtype)
    (median_TCC_HIT_RATE,median_time_ns,median_TCC_EA0_RDREQ , median_EA0_RDREQ_LEVEL) = parseCsvResults('res.csv_counter_collection.csv')
    
    with open(fileName, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([index,m, n, k, median_TCC_HIT_RATE, median_time_ns,median_TCC_EA0_RDREQ,median_EA0_RDREQ_LEVEL])



