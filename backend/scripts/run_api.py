import os
import sys
import subprocess

# Ensure we can import from optibatch modules if script is run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_server():
    """Launch the FastAPI server using Uvicorn."""
    print("Starting OptiBatch API Server...")
    print("API will be available at: http://127.0.0.0:8000")
    print("Interactive Swagger docs at: http://127.0.0.0:8000/docs\n")
    
    subprocess.run(["uvicorn", "optibatch.api.app:app", "--reload", "--port", "8000"])

if __name__ == "__main__":
    run_server()
