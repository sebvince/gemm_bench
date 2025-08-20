import json
import numpy as np

# Parse Json results from rocprofv3 to display L2Cache Hit rate per XCD
# To use with:
# PROFILER="rocprofv3  --pmc TCC_HIT,TCC_MISS --output-format json --stats --output-file res.json -- "
# Display L2 cache hit rate per XCD

def parseJsonResults(filename, M, N, K):
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
        
        time_ns = np.median(timings_ns)
        dispatch_index = timings_ns.index(time_ns)

        dispatch_records = dispatches[dispatch_index]["records"]
    
        hits = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == countersDict['TCC_HIT']]
        misses = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == countersDict['TCC_MISS']]
        EA_reqs = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == countersDict['TCC_EA0_RDREQ']]

        hits_xcc = [sum(hits[i:i+16]) for i in range(0, len(hits), 16)]
        misses_xcc = [sum(misses[i:i+16]) for i in range(0, len(misses), 16)]
        EA_reqs_xcc = [sum(EA_reqs[i:i+16]) for i in range(0, len(EA_reqs), 16)]
       
        for index in range(len(hits_xcc)):
            L2hitrate = 100.0*hits_xcc[index]/(hits_xcc[index]+misses_xcc[index])
            EA_reqs = EA_reqs_xcc[index]
            # print(f'Hit {L2hitrate} % - Reqs : {EA_reqs} - tcc_tag_stall : {tcc_tag_stall_xcc[index]}')
            print(f'L2HitRate {L2hitrate} % - EA Reqs : {EA_reqs}')
        
        print(f'Time : {time_ns/1e3} us')
        print("TFLOPS/s :", N*M*K*2/time_ns/1e3)


if __name__ == "__main__":
    M=8192
    N=32768
    K=2048
    dtype='f16'
    filename ='res.json_results.json'
   
    parseJsonResults(filename,M,N,K)