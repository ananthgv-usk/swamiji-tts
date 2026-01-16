import runpod
import json

POD_ID = "uryfa05fzanezn"  # New pod

try:
    pod = runpod.get_pod(POD_ID)
    print(json.dumps(pod, indent=2))
except Exception as e:
    print(e)
