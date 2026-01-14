import runpod
import time

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

# Terminate the broken pod
OLD_POD_ID = "0rice6j5w1okh9"
print(f"Terminating broken pod {OLD_POD_ID}...")
try:
    runpod.terminate_pod(OLD_POD_ID)
    print("✓ Old pod terminated")
except Exception as e:
    print(f"Warning: Could not terminate old pod: {e}")

time.sleep(5)

# Deploy fresh pod with CUDA 12.x compatible image
print("\nDeploying fresh H100 with CUDA 12.x support...")

pod = runpod.create_pod(
    name="Repo-JEPA-H100-CUDA12",
    image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",  # CUDA 12.x compatible
    gpu_type_id="NVIDIA H100 PCIe",
    gpu_count=1,
    volume_in_gb=100,
    container_disk_in_gb=25,
    ports="22/tcp,8888/tcp",
    env={
        "HF_TOKEN": HF_TOKEN or "",
        "PYTHONUNBUFFERED": "1"
    }
)

pod_id = pod['id']
print(f"\n✅ New pod created! ID: {pod_id}")
print("Waiting for pod to start...")

# Wait for pod to be ready
for i in range(60):
    time.sleep(5)
    pod_info = runpod.get_pod(pod_id)
    if pod_info and pod_info.get('runtime'):
        print("✓ Pod is running!")
        
        # Extract connection info
        runtime = pod_info['runtime']
        ports = runtime.get('ports', [])
        
        ssh_port = None
        jupyter_port = None
        ip = None
        
        for port in ports:
            if port.get('privatePort') == 22:
                ssh_port = port.get('publicPort')
                ip = port.get('ip')
            elif port.get('privatePort') == 8888:
                jupyter_port = port.get('publicPort')
        
        print("\n" + "="*60)
        print("🚀 H100 POD READY - CUDA 12.x")
        print("="*60)
        print(f"Pod ID: {pod_id}")
        if ip and ssh_port:
            print(f"SSH: ssh root@{ip} -p {ssh_port}")
        if ip and jupyter_port:
            print(f"Jupyter: http://{ip}:{jupyter_port}")
        print("="*60)
        
        print("\n📋 RUN THESE COMMANDS IN THE POD TERMINAL:")
        print("-" * 60)
        print(f"""
# 1. Verify GPU is working
nvidia-smi
python -c "import torch; print(f'CUDA: {{torch.cuda.is_available()}}'); print(f'GPU: {{torch.cuda.get_device_name(0)}}')"

# 2. Clone repo and install dependencies
git clone https://github.com/uddeshya-23/Repo-JEPA.git /Repo-JEPA
cd /Repo-JEPA
pip install -r requirements.txt

# 3. START TRAINING (this should work immediately!)
python -m src.train --epochs 10 --batch-size 64 --force-real-data
""")
        print("-" * 60)
        break
    
    if i % 6 == 0:
        print(f"Still waiting... ({i*5}s)")

if not pod_info or not pod_info.get('runtime'):
    print("Warning: Pod took too long to start. Check RunPod dashboard.")
