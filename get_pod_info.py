
import runpod
import os

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
if pod and pod.get('runtime'):
    runtime = pod['runtime']
    address = runtime.get('address')
    ports = runtime.get('ports', [])
    ssh_port = next((p['externalPort'] for p in ports if p['containerPort'] == 22), None)
    
    print(f"STATUS: RUNNING")
    print(f"ADDRESS: {address}")
    print(f"SSH_PORT: {ssh_port}")
else:
    print("STATUS: NOT_READY")
