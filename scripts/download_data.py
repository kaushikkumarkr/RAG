import os
import requests

DATA_DIR = "sample_data/large_dataset"

FILES = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "gpt4_technical_report.pdf": "https://arxiv.org/pdf/2303.08774.pdf",
    "llama_paper.pdf": "https://arxiv.org/pdf/2302.13971.pdf"
}

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    for filename, url in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        print(f"Downloading {filename} from {url}...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Saved {filename} ({len(response.content) / 1024 / 1024:.2f} MB)")
            else:
                print(f"❌ Failed to download {filename}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")

if __name__ == "__main__":
    download_data()
