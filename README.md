# Authentix: Voice Deepfake & Fraud Detection Engine 🎙️🛡️

Authentix is a full-stack, real-time audio forensics application designed to detect synthetic, AI-generated speech (deepfakes). 
Built to combat voice cloning fraud in telephony and VoIP environments, it processes audio streams through a dual-branch machine learning architecture and visualizes anomalies on a React-based mission control dashboard.

# 🚀 Features

>Real-Time Audio Ingestion:Upload static audio files (.wav, .flac, .mp3) or intercept live microphone streams via WebSocket.
>Dual-Branch ML Architecture:
pectral Analysis: Leverages Linear Frequency Cepstral Coefficients (LFCCs) for high-speed edge processing.
Deep Feature Extraction: Uses self-supervised learning (SSL) embeddings (like Wav2Vec 2.0) coupled with Graph Attention Networks (AASIST) for state-of-the-art anomaly detection.
>Telephony Codec Simulation: Robust against real-world signal degradation. The pipeline simulates G.711 (16kHz) and GSM (8kHz) compression to differentiate between network artifacts and vocoder glitches.
>Live Dashboard Visualization: Built with React and Tailwind CSS, featuring real-time waveforms, frequency spectrograms, and an escalating Fraud Probability Gauge.

# 🛠️ Tech Stack

Frontend: React.js (Vite), Tailwind CSS, Lucide React, Chart.js / D3.js.
Backend: Python, FastAPI, Uvicorn, WebSockets.
Machine Learning / DSP: PyTorch, Torchaudio, Librosa.

# ⚙️ Getting Started

Prerequisites
Ensure you have the following installed:
Node.js (v16+)
Python (v3.9+)
Git

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/voice-deepfake-detector.git
cd voice-deepfake-detector

2. Backend Setup (FastAPI & ML)

Create a virtual environment and install the Python dependencies.

# Create and activate a virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

Start the FastAPI server on port 5003:

uvicorn app:app --host 0.0.0.0 --port 5003 --reload

The API will be available at http://localhost:5003 and interactive docs at http://localhost:5003/docs.

# 3. Frontend Setup (React)

Open a new terminal split, leaving the backend running, and navigate to the client folder.

cd client
npm install
npm run dev

The dashboard will be available at http://localhost:5173.


📂 Project Structure

voice-deepfake-detector/

├── client/  

│   ├── src/

│   │   ├── components/  

│   │   └── App.jsx

│   └── package.json

├── models/ 

├── scripts/ 

│   └── preprocess.py 

├── app.py  

├── train.py

└── requirements.txt        


# 🧠 Model Training (Optional)

If you wish to train the model from scratch:

Download a dataset (e.g., ASVspoof 2019/2021 or the Audio Deepfake Detection Kaggle dataset).

Place the audio files in a local directory (e.g., ./data).

Update the data paths in scripts/preprocess.py and run it to extract features.

Run python train.py to begin the training loop.

# 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
