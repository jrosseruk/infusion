"""
Test HuggingFace authentication and setup.
"""
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, whoami

# Load .env file
load_dotenv()

try:
    # Get token from environment
    token = os.getenv("HF_TOKEN")

    if not token:
        print("\n" + "="*60)
        print("✗ HF_TOKEN not found in environment!")
        print("="*60)
        print("\nPlease create a .env file with your HuggingFace token:")
        print("1. Copy .env.example to .env:")
        print("   cp .env.example .env")
        print("\n2. Edit .env and add your token:")
        print("   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx")
        print("\n3. Get your token from: https://huggingface.co/settings/tokens")
        print("="*60)
        exit(1)

    # Test authentication
    print("Testing HuggingFace authentication...")
    api = HfApi(token=token)
    user_info = whoami(token=token)

    print("\n" + "="*60)
    print("✓ Authentication successful!")
    print("="*60)
    print(f"Username: {user_info['name']}")
    print(f"Email: {user_info.get('email', 'N/A')}")

    # Handle organizations (list of dicts)
    orgs = user_info.get('orgs', [])
    if orgs:
        org_names = [org.get('name', str(org)) for org in orgs]
        print(f"Organizations: {', '.join(org_names)}")
    else:
        print("Organizations: None")

    print("="*60)

    print("\nYour HuggingFace setup is ready!")
    print("\nYou can use repo IDs like:")
    print(f"  - {user_info['name']}/gpt-neo-28m-tinystories")
    print(f"  - {user_info['name']}/my-model-name")

    # Check if user has any orgs
    if orgs:
        print("\nOr use organization repos:")
        for org in orgs:
            org_name = org.get('name', 'unknown')
            print(f"  - {org_name}/gpt-neo-28m-tinystories")

    print("\n" + "="*60)
    print("Next steps:")
    print("1. Create a repo at https://huggingface.co/new")
    print("   OR let the training script auto-create it")
    print("2. Run training with --hf_repo_id YOUR_USERNAME/REPO_NAME")
    print("="*60)

except Exception as e:
    print("\n" + "="*60)
    print("✗ Authentication failed!")
    print("="*60)
    print(f"Error: {e}")
    print("\nPlease run: huggingface-cli login")
    print("And paste your token from: https://huggingface.co/settings/tokens")
    print("="*60)
