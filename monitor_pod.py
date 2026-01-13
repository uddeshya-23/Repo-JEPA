
import runpod
import os
import json

def get_env_var(name):
    try:
        with open('.env') as f:
            for line in f:
                if line.startswith(f"{name}="):
                    val = line.split('=', 1)[1].strip().strip("'").strip('"')
                    return val
    except:
        pass
    return None

API_KEY = get_env_var("RUNPOD")
runpod.api_key = API_KEY
POD_ID = "owpqqn7e0s371v"

print(f"Checking status for Pod: {POD_ID}")
try:
    pod = runpod.get_pod(POD_ID)
    print(f"Status: {pod.get('desiredStatus')}")
    
    # Check if there are any logs (some SDK versions support this)
    # If not, we might need to use another method
    # Let's try to list files if we can or just check runtime info
    runtime = pod.get('runtime', {})
    print(f"Runtime: {json.dumps(runtime, indent=2)}")

except Exception as e:
    print(f"Error fetching pod info: {e}")
