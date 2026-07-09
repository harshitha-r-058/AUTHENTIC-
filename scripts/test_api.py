import requests
import glob

api_url = "http://localhost:8000/api/analyze-audio"
test_files = glob.glob("generated_audio/**/*.wav", recursive=True)[:2]
test_files += glob.glob("ljspeech_dataset/authentic/*.wav")[:2]

for file_path in test_files:
    print(f"Testing {file_path}...")
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "audio/wav")}
        try:
            response = requests.post(api_url, files=files)
            print(response.json())
        except Exception as e:
            print(f"Error: {e}")
