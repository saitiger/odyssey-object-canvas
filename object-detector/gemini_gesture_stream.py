import json
import os
import time
import threading
import cv2
import asyncio
import websockets
import platform

from google import genai
from google.genai import types

# Lower-latency Flash models
MODEL_VISION = "gemini-2.5-flash-lite"
MODEL_TEXT = "gemini-2.5-flash-lite"

PROMPT_VISION = (
    "You are a real-time perception assistant. Detect common household objects, fruits, and everyday items that are "
    "clearly visible in the frame (e.g., phone, mug, bottle, book, pen, keys, wallet, apple, banana, orange, "
    "remote, clock, cap). Ignore clothing or accessories. "
    "Return a short, comma-separated list (3-6 items max), using plain lowercase nouns. "
    "If none are visible, say 'none'."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant that converts a short object list into a concise, safe, cinematic mid-stream "
    "prompt for Odyssey. Keep it short (1 sentence). Focus on the objects present, not a full scene reset."
)

SEND_INTERVAL = 3.0  # seconds (send every 3 seconds)
UPLOAD_LONG_EDGE = 240  # downscale to reduce latency
JPEG_QUALITY = 60  # lower quality for faster upload/infer
WS_HOST = "localhost"
WS_PORT = 8765
USE_LLM_REWRITE = False
USE_LLM_FILTER = True
DEDUPE_WINDOW_SEC = 10
# Tk UI must run on the main thread on macOS; default to disabled to avoid crashes.
ENABLE_PROMPT_UI = False


class PromptController:
    def __init__(self):
        self.lock = threading.Lock()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def set_system_prompt(self, text: str):
        with self.lock:
            self.system_prompt = text.strip() or DEFAULT_SYSTEM_PROMPT

    def get_system_prompt(self) -> str:
        with self.lock:
            return self.system_prompt


class GeminiWorker:
    def __init__(self, client: genai.Client, prompt_controller: PromptController):
        self.client = client
        self.prompt_controller = prompt_controller
        self.lock = threading.Lock()
        self.pending = None  # (frame, sent_ts)
        self.last_result = (None, "waiting...", None, None, None)  # (frame, text, latency_ms, ts, rewritten)
        self.last_seen = {}  # object -> last_seen_ts
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame, sent_ts):
        with self.lock:
            self.pending = (frame, sent_ts)

    def get_last(self):
        with self.lock:
            return self.last_result

    def stop(self):
        self.stop_flag = True
        self.thread.join(timeout=2)

    def _rewrite_prompt(self, label: str) -> str:
        if not USE_LLM_REWRITE:
            return label

        system_prompt = self.prompt_controller.get_system_prompt()
        user_prompt = (
            "Convert this object list into a short Odyssey mid-stream prompt. "
            "Return only the prompt sentence.\n\n"
            f"Objects: {label}"
        )

        try:
            response = self.client.models.generate_content(
                model=MODEL_TEXT,
                contents=[system_prompt, user_prompt],
            )
            text = (response.text or "").strip()
            return text or label
        except Exception:
            return label

    def _normalize_objects(self, text: str):
        parts = [p.strip().lower() for p in text.split(",")]
        cleaned = []
        for p in parts:
            if not p or p in ("none", "uncertain"):
                continue
            # keep simple noun tokens
            p = "".join(ch for ch in p if ch.isalnum() or ch in (" ", "-")).strip()
            if not p:
                continue
            cleaned.append(p)
        # de-dupe while preserving order
        seen = set()
        out = []
        for p in cleaned:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _filter_objects_llm(self, raw_text: str):
        if not USE_LLM_FILTER:
            return self._normalize_objects(raw_text)

        system_prompt = (
            "You filter object detections from a webcam. Keep only real, common household objects and fruits. "
            "Remove people, clothing, body parts, and vague terms. "
            "Return a JSON array of lowercase nouns, max 5 items."
        )
        user_prompt = f"Raw objects: {raw_text}\nReturn JSON array only."

        try:
            response = self.client.models.generate_content(
                model=MODEL_TEXT,
                contents=[system_prompt, user_prompt],
            )
            text = (response.text or "").strip()
            # basic JSON array parse without extra deps
            if text.startswith("```"):
                text = text.strip("` \n")
                if text.startswith("json"):
                    text = text[4:].strip()
            if text.startswith("[") and text.endswith("]"):
                items = json.loads(text)
                if isinstance(items, list):
                    return self._normalize_objects(", ".join([str(x) for x in items]))
        except Exception:
            pass

        return self._normalize_objects(raw_text)

    def _apply_cooldown(self, objects, now_ts):
        if not objects:
            return []
        allowed = []
        for obj in objects:
            last = self.last_seen.get(obj, 0)
            if now_ts - last >= DEDUPE_WINDOW_SEC:
                allowed.append(obj)
                self.last_seen[obj] = now_ts
        return allowed

    def _run(self):
        while not self.stop_flag:
            item = None
            with self.lock:
                item = self.pending
                self.pending = None

            if item is None:
                time.sleep(0.01)
                continue

            frame, sent_ts = item

            h, w = frame.shape[:2]
            scale = UPLOAD_LONG_EDGE / max(h, w)
            if scale < 1.0:
                resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                resized = frame

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue
            image_bytes = buf.tobytes()

            try:
                response = self.client.models.generate_content(
                    model=MODEL_VISION,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        PROMPT_VISION,
                    ],
                )
                label = (response.text or "").strip() or "uncertain"
            except Exception as e:
                label = f"error: {e}"

            rewritten = None
            if label and not label.lower().startswith("error"):
                now_ts = time.time()
                objects = self._filter_objects_llm(label)
                objects = self._apply_cooldown(objects, now_ts)
                if objects:
                    label = ", ".join(objects)
                    first = objects[0]
                    rewritten = f"add {first} to the canvas"
                else:
                    label = "none"

            latency_ms = int((time.time() - sent_ts) * 1000)
            ts = int(time.time() * 1000)

            with self.lock:
                self.last_result = (frame, label, latency_ms, ts, rewritten)


class WebsocketBroadcaster:
    def __init__(self):
        self.clients = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, message: str):
        if not self.clients:
            return
        await asyncio.gather(*[ws.send(message) for ws in list(self.clients)])


def run_prompt_window(controller: PromptController):
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title("Odyssey Prompt Controller")
    root.geometry("520x260")

    ttk.Label(root, text="System prompt for Odyssey mid-stream rewriting:").pack(padx=12, pady=(12, 6), anchor="w")

    text = tk.Text(root, height=8, wrap="word")
    text.insert("1.0", controller.get_system_prompt())
    text.pack(fill="both", expand=True, padx=12)

    status = ttk.Label(root, text="")
    status.pack(padx=12, pady=(6, 0), anchor="w")

    def apply_prompt():
        controller.set_system_prompt(text.get("1.0", "end").strip())
        status.config(text="Saved")
        root.after(1000, lambda: status.config(text=""))

    ttk.Button(root, text="Save Prompt", command=apply_prompt).pack(padx=12, pady=10, anchor="e")

    root.mainloop()


async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")

    client = genai.Client(api_key=api_key)
    prompt_controller = PromptController()

    if ENABLE_PROMPT_UI:
        if platform.system() == "Darwin":
            # Tk must be on main thread on macOS.
            raise RuntimeError(
                "ENABLE_PROMPT_UI is true, but Tk must run on the main thread on macOS. "
                "Set ENABLE_PROMPT_UI = False (default) or run UI in a separate process."
            )
        # Start prompt controller window in separate thread (non-macOS)
        threading.Thread(target=run_prompt_window, args=(prompt_controller,), daemon=True).start()

    worker = GeminiWorker(client, prompt_controller)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    broadcaster = WebsocketBroadcaster()
    server = await websockets.serve(broadcaster.handler, WS_HOST, WS_PORT)

    print(f"WebSocket server running on ws://{WS_HOST}:{WS_PORT}")
    print("Press q in the window to quit.")

    last_sent = 0.0
    last_frame_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = now - last_frame_time
            if dt > 0:
                fps = 1.0 / dt
            last_frame_time = now

            if now - last_sent >= SEND_INTERVAL:
                last_sent = now
                worker.submit(frame.copy(), now)

            processed_frame, label, latency_ms, ts, rewritten = worker.get_last()

            payload = {
                "label": label,
                "rewritten": rewritten or label,
                "latency_ms": latency_ms,
                "ts": ts,
            }
            await broadcaster.broadcast(json.dumps(payload))

            # Left panel: live video
            left = frame

            # Right panel: last processed frame + label
            if processed_frame is None:
                right = frame.copy()
                overlay_text = "waiting..."
                overlay_rewrite = ""
            else:
                right = processed_frame.copy()
                overlay_text = label
                overlay_rewrite = rewritten or ""

            cv2.rectangle(right, (0, 0), (right.shape[1], 110), (0, 0, 0), -1)
            cv2.putText(
                right,
                f"Objects: {overlay_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if overlay_rewrite:
                cv2.putText(
                    right,
                    f"Prompt: {overlay_rewrite[:80]}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                )
            if latency_ms is not None:
                cv2.putText(
                    right,
                    f"Latency: {latency_ms} ms",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            # Optional: FPS on left
            cv2.putText(
                left,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            side_by_side = cv2.hconcat([left, right])
            cv2.imshow("Object Detection (Left: Live | Right: Processed)", side_by_side)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.001)
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
