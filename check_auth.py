
import os
from huggingface_hub import HfApi
from pathlib import Path

def check_token():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    value = value.strip("'").strip('"')
                    os.environ[key] = value

    
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: No HF_TOKEN found in .env or environment.")
        return

    api = HfApi(token=token)
    try:
        user = api.whoami()
        print(f"✅ Success! Authenticated as: {user['name']} ({user['fullname']})")
        print(f"Full Auth Info: {user.get('auth', {})}")
        # Check for write access in the new format or old format
        is_write = user.get('auth', {}).get('accessToken', {}).get('role', '') == 'write'
        print(f"Token has explicit 'write' role: {is_write}")
    except Exception as e:

        print(f"❌ Authentication failed: {e}")

if __name__ == "__main__":
    check_token()
