import json
# To use with:
# PROFILER="rocprofv3  --pmc TCC_HIT,TCC_MISS --output-format json --stats --output-file res.json -- "
# Display L2 cache hit rate per XCD

with open('res.json_results.json', 'r') as file:
    data = json.load(file)

    
    dispatches = data['rocprofiler-sdk-tool'][0]['callback_records']['counter_collection']
    dispatch_records = dispatches[0]["records"]
    print("nb records", len(dispatch_records))

    hits = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == 3862]
    misses = [record['value'] for record in dispatch_records if record['counter_id']['handle'] == 3884]

    # for index in range(len(hits)):
    #     print(100.0*hits[index]/(hits[index]+misses[index]))

    hits_xcc = [sum(hits[i:i+16]) for i in range(0, len(hits), 16)]
    misses_xcc = [sum(misses[i:i+16]) for i in range(0, len(hits), 16)]

    for index in range(len(hits_xcc)):
        print(100.0*hits_xcc[index]/(hits_xcc[index]+misses_xcc[index]))
    
