# ps-2-001

Status: passed

Verified every update against single-process FP64 MBGD, including step 1 and step 20. Maximum parameter absolute difference: 5.55112e-17.

Workers never update weights; server weights local mean gradients by actual shard sample counts. Launcher cleanup and individual exit codes are in results.json.

See config.json, results.json and timings.csv for raw evidence.
