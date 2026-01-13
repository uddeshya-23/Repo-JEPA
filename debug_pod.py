
import runpod
import os
import json

# Function to load .env manually
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

pod = runpod.get_pod(POD_ID)
print(json.dumps(pod, indent=2))
