import json
import numpy as np
import sys
import os
import statistics
import csv

# Parse Json results from rocprofv3 to display L2Cache Hit rate per XCD
# To use with:
# PROFILER="rocprofv3  --pmc TCC_HIT,TCC_MISS --output-format json --stats --output-file res.json -- "
# Display L2 cache hit rate per XCD

def parseJsonResults(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        counters = data['rocprofiler-sdk-tool'][0]['counters']
        countersDict = {}
        for counter in counters:
            countersDict[counter['name']] = counter['id']['handle']

        required_keys = ['TCC_MISS', 'TCC_HIT']
        if not all(key in countersDict for key in required_keys):
            print(f'{",".join(required_keysOne)} missings')

        dispatches = data['rocprofiler-sdk-tool'][0]['callback_records']['counter_collection']

        #Pick median timing dispatch
        timings_ns = []
        for index in range(len(dispatches)):
            dispatch_data = dispatches[index]["dispatch_data"]
            timings_ns.append(float(dispatch_data["end_timestamp"])-float(dispatch_data["start_timestamp"]))
        
        sorted_indices = sorted(range(len(timings_ns)), key=lambda i: timings_ns[i])
        dispatch_index = sorted_indices[len(sorted_indices)//2]
        time_ns = timings_ns[dispatch_index]
    
        dispatch_records = dispatches[dispatch_index]["records"]
    
        hits = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == countersDict['TCC_HIT']]
        misses = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == countersDict['TCC_MISS']]
        # EA_reqs = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == countersDict['TCC_EA0_RDREQ']]

        hits_xcc = [sum(hits[i:i+16]) for i in range(0, len(hits), 16)]
        misses_xcc = [sum(misses[i:i+16]) for i in range(0, len(misses), 16)]
        # EA_reqs_xcc = [sum(EA_reqs[i:i+16]) for i in range(0, len(EA_reqs), 16)]
       
        for index in range(len(hits_xcc)):
            L2hitrate = 100.0*hits_xcc[index]/(hits_xcc[index]+misses_xcc[index])
            # EA_reqs = EA_reqs_xcc[index]
            print(f'L2HitRate {L2hitrate}')
        
        print(f'Time : {time_ns/1e3} us')

def parseCsvResults(filename):
    data = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        shape = None

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
        
        TCC_HIT_RATES = [value['TCC_HIT']/(value['TCC_MISS']+value['TCC_HIT']) for value in data.values()] 
        times_ns = [value['time'] for value in data.values()] 
        TCC_EA0_RDREQs = [value['TCC_EA0_RDREQ'] for value in data.values()] 
        TCC_EA0_RDREQ_LEVELs = [value['TCC_EA0_RDREQ_LEVEL'] for value in data.values()] 
        # Calculate the median
        median_TCC_HIT_RATE = statistics.median(TCC_HIT_RATES)
        median_time_ns = statistics.median(times_ns)
        median_TCC_EA0_RDREQ = statistics.median(TCC_EA0_RDREQs)
        median_EA0_RDREQ_LEVEL = statistics.median(TCC_EA0_RDREQ_LEVELs)
        return (median_TCC_HIT_RATE,median_time_ns,median_TCC_EA0_RDREQ,median_EA0_RDREQ_LEVEL)

if __name__ == "__main__":
    filename = sys.argv[1]
    _, file_extension = os.path.splitext(filename)
    if file_extension == ".csv":
        (median_TCC_HIT_RATE,median_time_ns,median_TCC_EA0_RDREQ , median_EA0_RDREQ_LEVEL) = parseCsvResults(filename)
        print("TCC_HIT_RATE:", median_TCC_HIT_RATE)
        print("EA Latency:", median_EA0_RDREQ_LEVEL/median_TCC_EA0_RDREQ)
        print("Time (ms):", median_time_ns/1e6)
    elif file_extension == ".json":
        parseJsonResults(filename)
