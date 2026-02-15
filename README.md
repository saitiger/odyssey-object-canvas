# Odyssey Object Canvas (Combined)

This repo contains:
- `object-detector/` — Gemini webcam object detection -> WebSocket feed (`ws://localhost:8765`)
- `odyssey-app/` — Odyssey UI that consumes the object feed and injects prompts midstream

## Run (separate processes)

Run **both** in separate terminals: (1) object detector, then (2) Odyssey app.

---

### 1) Object detector (WebSocket feed)

**macOS / Linux:**
```bash
cd "./object-detector"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export GEMINI_API_KEY="YOUR_KEY"
python gemini_gesture_stream.py
```

**Windows (PowerShell):**
```powershell
cd object-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:GEMINI_API_KEY = "YOUR_KEY"
python gemini_gesture_stream.py
```

> **Tip:** If `Activate.ps1` won’t run (execution policy error), run once:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`  
> Then try activating again.

> Replace `YOUR_KEY` with your [Google AI (Gemini) API key](https://aistudio.google.com/apikey). The detector will serve object detection at `ws://localhost:8765`.

---

### 2) Odyssey app

Same on all platforms:
```bash
cd odyssey-app
npm install
npm run dev
```

Then open the URL Vite prints (e.g. `http://localhost:5173`). Set your Odyssey API key in **Settings**, connect, then toggle **Object prompts** on.
