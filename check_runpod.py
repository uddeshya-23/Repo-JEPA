
import runpod
import os

api_key = "rpa_XK7HKSU59SZ9ZO06RR7JA3ASU8T3ZDE5FW22CZCVSjfvmsj" # User provided
runpod.api_key = api_key

print("Fetching available GPUs...")
gpu_types = runpod.get_gpu_types()
for gpu in gpu_types:
    if "H100" in gpu['id']:
        print(f"ID: {gpu['id']}, Name: {gpu['displayName']}")

print("\nFetching templates...")
# Since there's no direct get_templates in the simplified SDK docs often, 
# we might just try to find a standard one or look at the pod creation options.
# But usually we can just use the ID.
