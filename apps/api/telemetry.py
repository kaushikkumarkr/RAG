import os
from langfuse import Langfuse

def setup_telemetry():
    # Langfuse client automatically picks up environment variables:
    # LANGFUSE_SECRET_KEY
    # LANGFUSE_PUBLIC_KEY
    # LANGFUSE_HOST
    
    try:
        if os.getenv("LANGFUSE_PUBLIC_KEY"):
            langfuse = Langfuse()
            print(f"Langfuse initialized. Connected to {os.getenv('LANGFUSE_HOST')}")
            
            # Verify auth
            langfuse.auth_check()
        else:
            print("Langfuse credentials not found. Telemetry disabled.")
    except Exception as e:
        print(f"Failed to initialize Langfuse: {e}")
