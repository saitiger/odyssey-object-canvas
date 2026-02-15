# Odyssey Hackathon

This is a streamlined version of the Odyssey streaming demo.

## Behavior
- Only a **single starting prompt** is used to start the stream.
- **All follow-up prompts are disabled.**
- The only way to change the stream mid‑flight is via **gesture prompts** (WebSocket feed).

## Run

```bash
cd "/Users/sachin/Documents/New project/odyssey-hackathon"
npm install
npm run dev
```

Open the local URL printed by Vite.

## Gesture Feed

Start the local gesture stream in a separate terminal:

```bash
source ~/.venv-gemini/bin/activate
export GEMINI_API_KEY="YOUR_API_KEY"
python ~/gemini_gesture_stream.py
```

Toggle **Gesture prompts** ON in the header. Each event becomes:

```
User action: <label>
```

## Notes
- Follow-up prompt scheduling is disabled in `index.html`.
- Gesture feed WebSocket: `ws://localhost:8765`
