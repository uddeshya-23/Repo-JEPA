
import runpod
import time
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

# Configuration
API_KEY = get_env_var("RUNPOD")
HF_TOKEN = get_env_var("HF_TOKEN")

if not API_KEY:
    print("Error: RUNPOD API Key not found in .env")
    exit(1)

runpod.api_key = API_KEY

GPU_TYPE_ID = "NVIDIA H100 PCIe" # Common H100 ID on RunPod

def deploy_and_train():
    print(f"Deploying pod with {GPU_TYPE_ID}...")
    
    # Create pod
    try:
        # Note: image_name is the Docker image. 
        # For H100, we want a recent PyTorch image.
        pod = runpod.create_pod(
            name="Repo-JEPA-H100",
            image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
            gpu_type_id=GPU_TYPE_ID,
            gpu_count=1,
            volume_in_gb=50,
            container_disk_in_gb=20,
            ports="22/tcp,8888/tcp",
            env={
                "HF_TOKEN": HF_TOKEN,
                "PYTHONUNBUFFERED": "1"
            }
        )
        
        pod_id = pod['id']
        print(f"Pod created! ID: {pod_id}")
        
        print("Waiting for pod to be ready...")
        while True:
            pod_info = runpod.get_pod(pod_id)
            if pod_info and pod_info.get('runtime'):
                print("Pod is running!")
                break
            print("Still starting...")
            time.sleep(10)
            
        print("\n--- DEPLOYMENT SUCCESSFUL ---")
        print(f"Pod ID: {pod_id}")
        # Connection info
        runtime = pod_info.get('runtime', {})
        address = runtime.get('address')
        ports = runtime.get('ports', [])
        ssh_port = next((p['externalPort'] for p in ports if p['containerPort'] == 22), None)
        
        if address and ssh_port:
            print(f"SSH: ssh -p {ssh_port} root@{address}")
        
        print("\nSetup commands for the pod:")
        print(f"git clone https://github.com/uddeshya-23/Repo-JEPA.git && cd Repo-JEPA")
        print(f"pip install -r requirements.txt")
        print(f"python -m src.train --epochs 10 --batch-size 64")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    deploy_and_train()
