import json

# To use with:
# PROFILER="rocprofv3  --pmc TCC_HIT,TCC_MISS --output-format json --stats --output-file res.json -- "
# Display L2 cache hit rate per XCD

def parseJsonResults(filename, M, N, K):
    with open(filename, 'r') as file:
        data = json.load(file)
        dispatches = data['rocprofiler-sdk-tool'][0]['callback_records']['counter_collection']
        dispatch_index = 10
        dispatch_data = dispatches[dispatch_index]["dispatch_data"]
        time_ns = float(dispatch_data["end_timestamp"])-float(dispatch_data["start_timestamp"])
        dispatch_records = dispatches[dispatch_index]["records"]
        print("nb records", len(dispatch_records))

        hits = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == 3862]
        misses = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == 3884]

        hits_xcc = [sum(hits[i:i+16]) for i in range(0, len(hits), 16)]
        misses_xcc = [sum(misses[i:i+16]) for i in range(0, len(hits), 16)]

        for index in range(len(hits_xcc)):
            print(100.0*hits_xcc[index]/(hits_xcc[index]+misses_xcc[index]))
        
        print("Time : ", time_ns/1e3)
        print("TFLOPS/s :", N*M*K*2/time_ns/1e3)