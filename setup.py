import subprocess
import sys

def main():
    """
    Finalizes the environment setup by installing performance-critical CUDA kernels.
    Note: Core dependencies are automatically synced by 'uv run' before this script executes.
    """
    kernels = ["flash-attn", "causal-conv1d", "flash-linear-attention"]
    
    print("\n" + "="*50)
    print("🚀 QWEN-OCR SETUP: Installing Performance Kernels")
    print("="*50)

    
    try:
        # We use --force-reinstall and --no-cache-dir to ensure we fix "broken" binaries
        # that Unsloth might have detected in the existing environment.
        subprocess.check_call([
            "uv", "pip", "install", 
            *kernels, 
            "--no-build-isolation",
            "--force-reinstall",
            "--no-cache-dir"
        ])
        print("\n✅ Kernels re-installed and optimized successfully!")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to install kernels. Error code: {e.returncode}")
        print("Note: This step requires a GPU with CUDA headers and a compatible C++ compiler.")
        sys.exit(1)

if __name__ == "__main__":
    main()
