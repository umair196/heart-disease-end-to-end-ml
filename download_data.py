import os
import requests

URLS = [
    "https://raw.githubusercontent.com/kinir/catboost-with-pipelines/master/cardiovascular-disease-dataset/original/cardio_train.csv",
    "https://raw.githubusercontent.com/caravanuden/cardio/master/cardio_train.csv",
]

def main():
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/cardio_train.csv"

    if os.path.exists(output_path):
        print("✅ Dataset already exists.")
        return

    for url in URLS:
        try:
            print(f"Trying {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(r.content)
            print("✅ Dataset downloaded successfully!")
            return
        except Exception as e:
            print("❌ Failed:", e)

    print("❌ Could not download dataset.")

if __name__ == "__main__":
    main()