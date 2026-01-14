
import runpod
import os
import time

# Configuration
GPU_TYPE_ID = "NVIDIA H100 PCIe"
REPO_URL = "https://github.com/uddeshya-23/Repo-JEPA.git"

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

RUNPOD_API_KEY = get_env_var("RUNPOD")
HF_TOKEN = get_env_var("HF_TOKEN")

if not RUNPOD_API_KEY:
    print("Error: RUNPOD API key not found in .env")
    exit(1)

runpod.api_key = RUNPOD_API_KEY

print(f"Deploying {GPU_TYPE_ID} pod for REAL-DATA training...")

# Add setup commands
setup_commands = f"""
export HF_TOKEN="{HF_TOKEN or ""}"
apt-get update && apt-get install -y git-lfs
git clone {REPO_URL} /Repo-JEPA
cd /Repo-JEPA
pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0
# Start training with real data enforcement and optimized batch size
# We use 128 batch size to fully utilize H100 80GB VRAM
python -m src.train --epochs 10 --batch-size 128 --force-real-data
"""

pod = runpod.create_pod(
    name="Repo-JEPA-Real-Train",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
    gpu_type_id=GPU_TYPE_ID,
    gpu_count=1,
    volume_in_gb=100,
    container_disk_in_gb=20,
    ports="22/tcp,8888/tcp",
    env={
        "HF_TOKEN": HF_TOKEN or "",
        "PYTHONUNBUFFERED": "1"
    }
)

pod_id = pod['id']
print(f"Pod created! ID: {pod_id}")
print("Waiting for pod to initialize...")

while True:
    pod_info = runpod.get_pod(pod_id)
    if pod_info and pod_info.get('runtime'):
        print("Pod is running!")
        break
    time.sleep(5)

print("\n--- Pod Connection Info ---")
print(f"Pod ID: {pod_id}")
if 'runtime' in pod_info:
    address = pod_info['runtime']['address']
    ssh_port = [p['external_port'] for p in pod_info['runtime']['ports'] if p['container_port'] == 22][0]
    print(f"SSH: ssh root@{address} -p {ssh_port}")
    print(f"Jupyter: http://{address}:8888")

print("\n--- Commands to run inside the pod ---")
print(setup_commands)
print("\nCopy and paste the above commands into the Pod's terminal (Web Terminal or SSH) to start training.")
