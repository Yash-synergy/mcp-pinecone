#!/usr/bin/env python3
import json
import sys
from datetime import datetime

def get_current_time():
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    # Read request from stdin
    request_line = sys.stdin.readline()
    request = json.loads(request_line)
    
    # Validate request
    if request.get("type") != "tool":
        print(json.dumps({"error": "Invalid request type"}))
        return
    
    payload = request.get("payload", {})
    action = payload.get("action")
    
    if action != "get-time":
        print(json.dumps({"error": f"Invalid action: {action}"}))
        return
    
    # Process the request
    result = get_current_time()
    
    # Return the result
    response = {
        "result": result
    }
    
    print(json.dumps(response))
    sys.stdout.flush()

if __name__ == "__main__":
    main() 