import runpod
import os
import sys

runpod.api_key = os.getenv("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY")
POD_ID = os.getenv("POD_ID", "YOUR_POD_ID")

HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")

def exec_cmd(cmd_list):
    query = """
    mutation PodExec($podId: String!, $command: [String!]!) {
        podExec(input: {podId: $podId, command: $command}) {
            stdout
            stderr
        }
    }
    """
    try:
        result = runpod.api.graphql.run_graphql_query(query, {"podId": POD_ID, "command": cmd_list}, api_key=runpod.api_key)
        if 'errors' in result:
             print("Error:", result['errors'])
             return None
        return result['data']['podExec']
    except Exception as e:
        print(f"Exception: {e}")
        return None

def write_file_to_pod(local_path, remote_name):
    print(f"Deploying {local_path} to {remote_name}...")
    with open(local_path, "r") as f:
        content = f.read()

    # Prepend HF_TOKEN export if it's the setup script
    if remote_name == "setup.sh":
         content = f"export HF_TOKEN={HF_TOKEN}\n" + content

    # Escape single quotes for bash heredoc
    # We use EOF_DEPLOY delimiter
    # We need to construct: cat > remote_name <<'EOF_DEPLOY' \n content \n EOF_DEPLOY
    
    # Actually, sending huge string via args list might be limited or parsed weirdly.
    # But usually GraphQL accepts strings.
    # IMPORTANT: The command is a list of strings [ "bash", "-c", "..." ]
    
    # We need to be careful with existing single quotes in content.
    # Strategy: Replace ' with '\'' inside the content for the sh -c '...' encapsulation?
    # No, heredoc inside '...' is tricky.
    
    # Better strategy: Use base64
    import base64
    b64_content = base64.b64encode(content.encode()).decode()
    
    # proper command: echo "base64" | base64 -d > filename
    cmd = ["bash", "-c", f"echo '{b64_content}' | base64 -d > {remote_name}"]
    
    res = exec_cmd(cmd)
    if res:
        print("STDOUT:", res['stdout'])
        print("STDERR:", res['stderr'])

def main():
    write_file_to_pod("setup_runpod_orpheus.sh", "setup.sh")
    write_file_to_pod("train_orpheus_500steps.py", "train_orpheus.py")
    
    print("Running setup.sh...")
    # Run setup
    res = exec_cmd(["bash", "setup.sh"])
    if res:
        print("Setup STDOUT:", res['stdout'])
        print("Setup STDERR:", res['stderr'])
        
    print("Files deployed.")

if __name__ == "__main__":
    main()
