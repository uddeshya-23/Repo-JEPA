
import runpod
import time
import os

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

# 1. Terminate any previous broken pods
print("Checking for existing Repo-JEPA pods...")
pods = runpod.get_pods()
for p in pods:
    if "Repo-JEPA" in p.get('name', ''):
        print(f"Terminating old pod: {p['id']} ({p['name']})")
        try:
            runpod.terminate_pod(p['id'])
        except:
            pass

time.sleep(5)

# 2. Deploy with NVIDIA NGC Image (Highly optimized for H100)
# We use the latest 24.12 release which is built for the latest H100 architectures and drivers.
GPU_TYPE_ID = "NVIDIA H100 PCIe"
IMAGE_NAME = "nvcr.io/nvidia/pytorch:24.12-py3"

print(f"Deploying {GPU_TYPE_ID} with official NVIDIA NGC image...")
print(f"Image: {IMAGE_NAME}")

pod = runpod.create_pod(
    name="Repo-JEPA-Final-H100",
    image_name=IMAGE_NAME,
    gpu_type_id=GPU_TYPE_ID,
    gpu_count=1,
    volume_in_gb=100,
    container_disk_in_gb=50,
    ports="22/tcp,8888/tcp",
    env={
        "HF_TOKEN": HF_TOKEN or "",
        "PYTHONUNBUFFERED": "1"
    }
)

pod_id = pod['id']
print(f"Pod created! ID: {pod_id}")
print("Waiting for pod initialization (NGC images are large, may take 2-3 mins)...")

while True:
    pod_info = runpod.get_pod(pod_id)
    if pod_info and pod_info.get('runtime'):
        print("Pod is running!")
        break
    time.sleep(10)

# Setup connection info
runtime = pod_info['runtime']
address = runtime['address']
ssh_port = [p['external_port'] for p in runtime['ports'] if p['container_port'] == 22][0]
jupyter_port = [p['external_port'] for p in runtime['ports'] if p['container_port'] == 8888][0]

print("\n" + "="*50)
print("🚀 STABLE H100 POD DEPLOYED")
print("="*50)
print(f"Pod ID: {pod_id}")
print(f"SSH: ssh root@{address} -p {ssh_port}")
print(f"Jupyter: http://{address}:{jupyter_port}")
print("="*50)

# Commands to run
print("\n📋 ONE-SHOT TRAINING COMMANDS:")
print("-" * 50)
print(f"""
# 1. VERIFY GPU FIRST
nvidia-smi
python -c "import torch; print(f'CUDA available: {{torch.cuda.is_available()}}'); print(f'Device: {{torch.cuda.get_device_name(0)}}')"

# 2. CLONE AND SETUP (Using pre-installed PyTorch in NGC image)
git clone https://github.com/uddeshya-23/Repo-JEPA.git /Repo-JEPA
cd /Repo-JEPA

# NOTE: Do NOT install torch/torchvision from requirements.txt to avoid polluting NGC environment
# We only install the missing utility libraries
pip install transformers datasets accelerate einops sentencepiece scikit-learn wandb matplotlib seaborn

# 3. START TRAINING
python -m src.train --epochs 10 --batch-size 128 --force-real-data
""")
print("-" * 50)
