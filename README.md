# 🚒 AdjustLeads POC

A real-time emergency response proof-of-concept built with Streamlit. It continuously streams audio from a broadcast URL, processes and levels it, saves clips to a temporary directory, transcribes them with OpenAI’s Whisper, and extracts structured insights via GPT-4.

---

## 🔍 Features

- **Live audio streaming** from a configurable URL  
- **Automatic chunking** (configurable duration & bitrate)  
- **Silence detection & loudness leveling** with `pydub`, `librosa`, and `scipy`  
- **Temporary storage** under `/tmp/` for Streamlit Cloud compatibility  
- **Whisper-1 transcription** via OpenAI Audio API  
- **GPT-4-powered insight extraction** (summary, locations, priority, recommendations, etc.)  
- **Interactive Streamlit UI** for start/pause/stop, playback, and transcript display  

---

## 🚀 Quick Start

1. **Clone this repo**  
   ```bash
   git clone https://github.com/your-org/AdjustLeads-POC.git
   cd AdjustLeads-POC


2. Create & activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate

```

3. Install dependencies

``` bash
pip install -r requirements.txt
```
4. Environment variables
Create a .env file in the project root:

dotenv

OPENAI_API_KEY=sk-your_openai_key_here


Run locally

``` bash

streamlit run app.py
```
⚙️ Configuration
All core settings live at the top of app.py:

Variable	Description	Default
STREAM_URL	Broadcast MP3 stream URL	Your feed URL
CHUNK_SEC	Seconds per audio chunk	30
KBPS	Target bitrate (kbps)	16
MIN_DBFS	Minimum loudness threshold (dBFS) for valid chunks	-50
RETRY_TIMEOUT	Total seconds to retry fetching a chunk	40

—— Temporary directories (/tmp/…) are auto-created on each run so no local cleanup is needed.

🖥️ Deployment on Streamlit Cloud
Push your changes to GitHub.

On Streamlit Cloud, New app → select this repo.

Set your Environment variables (e.g. OPENAI_API_KEY) in the app settings.

Deploy. Streamlit Cloud will run under /tmp/ so audio/transcripts remain ephemeral.

📂 Project Structure
```bash

├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .env                  # Your OpenAI API key (not checked in)
└── .streamlit/           # Streamlit configuration (optional)
```
🛠️ Troubleshooting
“Permission denied” errors
→ Ensure you’re writing only under /tmp/ on Streamlit Cloud (handled by default).

Missing transcripts
→ Check your OPENAI_API_KEY and network connectivity.

High memory usage
→ Lower CHUNK_SEC or KBPS to reduce buffer size.

🤝 Contributing
Fork this repo

Create a feature branch (git checkout -b feature/xyz)

Commit your changes & push (git push origin feature/xyz)

Open a Pull Request

📄 License
This project is licensed under the MIT License.