# Odyssey Object Canvas (Combined)

This repo contains:
- `object-detector/` — Gemini webcam object detection -> WebSocket feed (`ws://localhost:8765`)
- `odyssey-app/` — Odyssey UI that consumes the object feed and injects prompts midstream

## Run (separate processes)

### 1) Object detector (WebSocket feed)
```bash
cd "./object-detector"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install google-genai opencv-python websockets
export GEMINI_API_KEY="YOUR_KEY"
python gemini_gesture_stream.py
```

### 2) Odyssey app
```bash
cd "./odyssey-app"
npm install
npm run dev
```

Open the app, set your Odyssey API key in Settings, connect, then toggle **Object prompts** on.
