import runpod
import time
import sys

# GPU Priorities
GPU_TYPES = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A40"
]

POD_NAME = "orpheus-training"
IMAGE = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"

def create_pod():
    user_api_key = runpod.api_key
    if not user_api_key:
        # try loading from config if not set in env (which simple script might miss if env var not passed)
        # But earlier CLI config should have saved it to ~/.runpod/config.toml which lib reads
        pass

    for gpu in GPU_TYPES:
        print(f"Attempting to launch pod with {gpu}...")
        try:
            pod = runpod.create_pod(
                name=POD_NAME,
                image_name=IMAGE,
                gpu_type_id=gpu,
                gpu_count=1,
                volume_in_gb=50, # Ensure enough space for model
                container_disk_in_gb=50,
                cloud_type="COMMUNITY" # Often higher availability
            )
            print(f"Success! Pod created: {pod['id']}")
            return pod['id']
        except Exception as e:
            print(f"Failed to create with {gpu}: {e}")
            time.sleep(1)
    
    return None

if __name__ == "__main__":
    pod_id = create_pod()
    if pod_id:
        print(f"POD_ID:{pod_id}")
        
        # Wait for running
        print("Waiting for pod to become RUNNING...")
        while True:
            try:
                pod_info = runpod.get_pod(pod_id)
                status = pod_info.get('desiredStatus', 'UNKNOWN') # or 'runtime'/'status' depending on API
                # Correct field checking
                if pod_info.get('runtime', {}).get('uptimeInSeconds', 0) > 0 and pod_info.get('desiredStatus') == "RUNNING":
                     print("Pod is RUNNING!")
                     break
            except:
                pass
            time.sleep(5)
    else:
        print("Could not provision any pod.")
        sys.exit(1)
