import requests
import time
import sys
import os

BASE_URL = "http://localhost:8000"

def log(msg):
    print(f"[TEST] {msg}")

def check_health(retries=10, delay=5):
    log("Checking API Health...")
    for i in range(retries):
        try:
            resp = requests.get(f"{BASE_URL}/health")
            if resp.status_code == 200:
                log(f"Health OK: {resp.json()}")
                return True
        except Exception as e:
            log(f"Attempt {i+1}/{retries} - Health Check Failed: {e}")
        time.sleep(delay)
    return False

def ingest_file(path):
    log(f"Ingesting file: {path}")
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "text/plain")}
        resp = requests.post(f"{BASE_URL}/ingest/file", files=files)
        if resp.status_code == 200:
            log(f"Ingest OK: {resp.json()}")
            return True
        else:
            log(f"Ingest Failed: {resp.text}")
            return False

def ask_agent(question):
    log(f"Asking Agent: '{question}'")
    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/agent/ask", json={"question": question}, timeout=120)
        duration = time.time() - start
        
        if resp.status_code == 200:
            result = resp.json()
            log(f"Agent Response ({duration:.2f}s): {result['answer']}")
            return result['answer']
        else:
            log(f"Agent Request Failed: {resp.text}")
            return None
    except Exception as e:
        log(f"Agent Request Error: {e}")
        return None

def main():
    if not check_health():
        sys.exit(1)

    # 1. Ingest
    if not ingest_file("sample_data/project_omega.txt"):
        sys.exit(1)

    # 2. Ask simple question to verify Agent can use SearchTool
    answer = ask_agent("Who led Project Omega?")
    if "Elena Vance" in answer or "Vance" in answer:
        log("PASS: Retrieved correct leader.")
    else:
        log("FAIL: Did not retrieve correct leader.")

    # 3. Ask complex question to verify Agent reasoning (if applicable, though simple Agent might just search)
    answer_complex = ask_agent("What was the budget of Project Omega and when did it start?")
    if "500 trillion" in answer_complex and "2025" in answer_complex:
        log("PASS: Retrieved budget and start date.")
    else:
        log("FAIL: Missing key details in complex answer.")

if __name__ == "__main__":
    main()
