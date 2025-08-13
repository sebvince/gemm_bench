import csv
import sys
import statistics
import re

def parseMatrixShape(dispatchName):
    match = re.search(r'matmul_like_(\d+)x(\d+)x(\d+)_f', dispatchName)
    if match:
        m, n, k = match.groups()
        return (int(m), int(n), int(k))

        
data = {}
with open(sys.argv[1], newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    shape = None

    print("*************************")
    for row in reader:
        correlation_id = int(row['Correlation_Id'])
        counter_name = row['Counter_Name']
        counter_value = float(row['Counter_Value'])
        start = float(row['Start_Timestamp'])
        end = float(row['End_Timestamp'])
        time_us = end - start

        if correlation_id not in data:
            data[correlation_id] = {}
            data[correlation_id]['time'] = time_us

        data[correlation_id][counter_name]=counter_value
        if not shape:
            shape = parseMatrixShape(row['Kernel_Name'])

    print("*************************")   
    TCC_HIT_RATES = [value['TCC_HIT']/(value['TCC_MISS']+value['TCC_HIT']) for value in data.values()] 
    times_ns = [value['time'] for value in data.values()] 
    # Calculate the median
    median_TCC_HIT_RATE = statistics.median(TCC_HIT_RATES)
    median_time_ns = statistics.median(times_ns)

    print("TCC_HIT_RATE:", median_TCC_HIT_RATE)
    print("TFLOPS:", shape[0]*shape[1]*shape[2]*2/median_time_ns/1e3)
    print("TIME:", median_time_ns)
    print("SHAPE:", shape)
    
    print("*************************")
        