import csv
import os
import sys
import argparse
from generate_solution import generate_files
from rocprofResultsParser import parseJsonResults,parseCsvResults
from single_test import compile,run

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

def updateRow(fileName, new_row):
    hasInserted = False
    rows = []
    if os.path.exists(fileName):
        with open(fileName, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                # Check if this row matches M, N, K
                if int(row[0]) == new_row[0] and int(row[1]) == new_row[1] and int(row[2]) == new_row[2]:
                    rows.append(new_row)
                    hasInserted = True
                else:
                    rows.append(row)

    if not hasInserted:
        rows.append(new_row)

    # Write the updated rows back to the file
    with open(fileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['M', 'N', 'K', 'TCC_HIT_RATE', 'time_ns','TCC_EA0_RDREQ','EA0_RDREQ_LEVEL'])
        writer.writerows(rows)
        

# fileName = 'summary'

# Options


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--startId', type=int, required=False, default= 0, help='Start id for benchmark')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic blocked shapes')
    parser.add_argument('--dtype', type=str, required=False, default="f16", help='Data type')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    
    tileSize = 256
    isStatic = not args.dynamic
    print(f"StartId: {args.startId}")
    print(f"Dynamic: {args.dynamic}")
    print(f"dtype: {args.dtype}")
    print(f"Outdir: {args.outdir}")

    fileName = args.outdir+ '/summary'
    fileName = fileName + '_static' if not args.dynamic else fileName + '_dynamic'
    fileName += f'_{args.dtype}'
    fileName += '.csv'
    # startId = getLastId(fileName) + 1
    print("Output to " + fileName)

    nb_tests = len(test_cases)
    # new_row = (131072,16640,4096,0.41144592064316166,31486707.5,653892473.0,236325697624.0)
    # updateRow("test.csv", new_row)

    # exit()
    for index in range(args.startId,nb_tests):
        (m,n,k) = test_cases[index]
        print(f'Matmul_{m}_{n}_{k} : {index}')
        generate_files(m,n,k,args.dtype,isStatic,tileSize)
        compile()
        run(m,n,k,args.dtype)
        # (median_TCC_HIT_RATE,median_time_ns,median_TCC_EA0_RDREQ , median_EA0_RDREQ_LEVEL) = parseCsvResults('res.csv_counter_collection.csv')
        new_row = parseCsvResults('res.csv_counter_collection.csv')
        updateRow(fileName, (m,n,k) + new_row)
    # with open(fileName, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([index,m, n, k, median_TCC_HIT_RATE, median_time_ns,median_TCC_EA0_RDREQ,median_EA0_RDREQ_LEVEL])
    

if __name__ == "__main__":
    main()

# isStatic = sys.argv[1]=='1'
# dtype = sys.argv[2]

# print("isStatic:", isStatic)

# fileName = fileName + '_static' if isStatic else fileName + '_dynamic'
# fileName += f'_{dtype}'
# fileName += '.csv'
# startId = getLastId(fileName)+1
# print("Output to " + fileName)



# for index in range(startId,nb_tests):
#     (m,n,k) = test_cases[index]
#     print(f'Matmul_{m}_{n}_{k} : {index}')
#     generate_files(m,n,k,dtype,isStatic,tileSize)
#     compile(transposedReorder)
#     run(m,n,k,dtype)
#     (median_TCC_HIT_RATE,median_time_ns,median_TCC_EA0_RDREQ , median_EA0_RDREQ_LEVEL) = parseCsvResults('res.csv_counter_collection.csv')
    
#     with open(fileName, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([index,m, n, k, median_TCC_HIT_RATE, median_time_ns,median_TCC_EA0_RDREQ,median_EA0_RDREQ_LEVEL])



