import json
import os
import time
import urllib.request
from datetime import datetime, timedelta

def check_for_updates(current_version: str, repo: str = "gitmichaelqiu/MLTabularPipelines"):
    """
    Checks for the latest release on GitHub and compares versions.
    Limited to once every 24 hours.
    """
    # Use a hidden file in the library directory to store the last check timestamp
    # This works because the library is usually copied into projects locally.
    lib_dir = os.path.dirname(os.path.dirname(__file__))
    last_check_file = os.path.join(lib_dir, ".last_update_check")
    
    now = datetime.now()
    
    if os.path.exists(last_check_file):
        try:
            with open(last_check_file, "r") as f:
                data = json.load(f)
                last_check_time = datetime.fromisoformat(data.get("last_check", "2000-01-01"))
        except Exception:
            last_check_time = datetime.min
    else:
        last_check_time = datetime.min

    # Only check once every 24 hours
    if now < last_check_time + timedelta(hours=24):
        return

    try:
        # GitHub API for latest release
        url = f"https://api.github.com/repos/{repo}/releases/latest"
        
        # User-Agent is required by GitHub
        headers = {"User-Agent": "mltabpipe-update-checker"}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            latest_version = data.get("tag_name", "").lstrip("v")
            
            if latest_version and latest_version != current_version:
                print(f"\n" + "*" * 50)
                print(f"  [mltabpipe] A new version is available: {latest_version} (Current: {current_version})")
                print(f"  Download it here: https://github.com/{repo}/releases")
                print("*" * 50 + "\n")
        
        # Save timestamp
        with open(last_check_file, "w") as f:
            json.dump({"last_check": now.isoformat()}, f)
            
    except Exception:
        # Silently fail if there's no internet or other API issues
        pass
