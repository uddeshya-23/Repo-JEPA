
import runpod
import os
import json

def get_env_var(name):
    try:
        with open('.env') as f:
            for line in f:
                if line.startswith(f"{name}="):
                    val = line.split('=', 1).strip().strip("'").strip('"')
                    return val
    except:
        pass
    return None

# Wait, the line split above has an error. fixing it.
def get_key():
    with open('.env') as f:
        for line in f:
            if 'RUNPOD=' in line:
                return line.split('RUNPOD=')[1].split()[0].strip().strip("'").strip('"')
    return None

runpod.api_key = get_key()
pod_id = "0rice6j5w1okh9"
pod_info = runpod.get_pod(pod_id)
print(json.dumps(pod_info, indent=2))
