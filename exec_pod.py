import runpod
import sys

import os

runpod.api_key = os.getenv("RUNPOD_API_KEY", "YOUR_API_KEY")
POD_ID = os.getenv("POD_ID", "YOUR_POD_ID")

def exec_command(cmd_list):
    # Construct command string representation manually for GraphQL
    # cmd_list is list of strings. JSON dump it.
    import json
    cmd_str = json.dumps(cmd_list)
    
    query = f"""
    mutation {{
        podExec(input: {{podId: "{POD_ID}", command: {cmd_str}}}) {{
            stdout
            stderr
        }}
    }}
    """
    
    try:
        # run_graphql_query(query, api_key)
        result = runpod.api.graphql.run_graphql_query(query, api_key=runpod.api_key)
        if 'errors' in result:
             print("GraphQL Errors:", result['errors'])
        else:
             data = result['data']['podExec']
             print("STDOUT:", data['stdout'])
             if data['stderr']:
                 print("STDERR:", data['stderr'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec_command(sys.argv[1:])
    else:
        print("Usage: python3 exec_pod.py cmd arg1 arg2 ...")
