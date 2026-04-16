import json
import textwrap

VIDEO_SOURCES_CODE = """
VIDEO_SOURCES = {
    "Music & Study": [
        {"name": "Lofi Girl", "url": "https://www.youtube.com/watch?v=jfKfPfyJRdk"},
        {"name": "Monstercat Silk", "url": "http://www.youtube.com/watch?v=WsDyRAPFBC8"},
        {"name": "Chillhop Music", "url": "http://www.youtube.com/watch?v=7NOSDKb0HlU"},
    ],
    "Nature & Animals": [
        {"name": "Explore.org Puppy Cam", "url": "http://www.youtube.com/watch?v=h-Z0wCdD3dI"},
        {"name": "iPanda", "url": "http://www.youtube.com/watch?v=9LvjI3NelAU"},
        {"name": "NamibiaCam", "url": "https://www.youtube.com/watch?v=ydYDqZQpim8"},
        {"name": "AfriCam", "url": "https://www.youtube.com/watch?v=qpukdDslCjk"},
        {"name": "Wildlife In The Forest", "url": "https://www.youtube.com/watch?v=F0GOOP82094"},
    ],
    "News & Finance": [
        {"name": "Sky News", "url": "http://www.youtube.com/watch?v=YDvsBbKfLPA"},
        {"name": "Al Jazeera English", "url": "http://www.youtube.com/watch?v=YDvsBbKfLPA"},
        {"name": "Bloomberg Television", "url": "http://www.youtube.com/watch?v=iEpJwprxDdk"},
    ],
    "Space & Science": [
        {"name": "NASA Live", "url": "http://www.youtube.com/watch?v=m3kR2KK8TEs"},
        {"name": "Blooket Live", "url": "http://www.youtube.com/watch?v=M5OKNwczOP8"},
    ],
    "Urban & Transprot": [
        {"name": "La Plata", "url": "https://www.youtube.com/watch?v=X-ir2KfXMX0"},
        {"name": "Tokyo Streets", "url": "https://www.youtube.com/watch?v=L6wO1-U2RTY"},
    ],
}
""".strip()

FETCHER_CODE = """
import os
import time
import tempfile
import base64
import subprocess
import cv2
import numpy as np
import pytchat
from typing import List, Tuple

try:
    from kaggle_benchmarks.content_types.videos import VideoContent
except ImportError:
    import abc
    class VideoContent(abc.ABC):
        @property
        @abc.abstractmethod
        def url(self) -> str: ...
        @property
        @abc.abstractmethod
        def mime_type(self) -> str: ...
        def get_payload(self) -> list[dict[str, str | dict[str, str]]]:
            return [{"type": "image_url", "image_url": {"url": self.url}}]

class StreamVideoContent(VideoContent):
    def __init__(self, frames: List[np.ndarray], fps: float):
        self.frames = frames
        self.fps = fps

    @property
    def url(self) -> str:
        return ""

    @property
    def mime_type(self) -> str:
        return "video/mp4"
        
    def get_payload(self) -> list[dict[str, str | dict[str, str]]]:
        # Override to return frames as a sequence of images to prevent 400 errors from OpenAI-like APIs
        payload = []
        for frame in self.frames:
            # Compress to a reasonable JPEG size to save tokens
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_frame = base64.b64encode(buffer).decode('utf-8')
            payload.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"}
            })
        return payload

class StreamFetcher:
    def __init__(self, url: str, fps: int = 1):
        self.url = url
        self.fps = fps
        try:
            self.video_id = url.split("v=")[1].split("&")[0]
        except (IndexError, AttributeError):
            self.video_id = url.split("/")[-1]
        self.chat = None
        self.temp_dir = tempfile.mkdtemp()

    def start(self):
        self.chat = pytchat.create(video_id=self.video_id)

    def get_data_window(self, duration_sec: int) -> Tuple[List[str], StreamVideoContent]:
        start_time = time.time()
        chat_messages = []
        frames = []
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
            temp_path = tf.name

        try:
            # Extract raw HLS stream URL first for stability
            stream_url = subprocess.check_output(
                ['yt-dlp', '-g', '-f', 'best[height<=360]', self.url], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            cmd = [
                'ffmpeg', '-y',
                '-i', stream_url,
                '-t', str(duration_sec),
                '-c', 'copy',
                '-movflags', '+faststart',
                '-loglevel', 'error',
                temp_path
            ]
            process = subprocess.Popen(cmd)
            
            while time.time() - start_time < duration_sec:
                if self.chat and self.chat.is_alive():
                    for c in self.chat.get().sync_items():
                        chat_messages.append(f"{c.author.name}: {c.message}")
                time.sleep(0.1)

            process.wait(timeout=10)
            
        except Exception as e:
            print(f"Warning: Could not fetch video properly ({e})")
        
        # Final chat drain
        if self.chat and self.chat.is_alive():
            for c in self.chat.get().sync_items():
                chat_messages.append(f"{c.author.name}: {c.message}")

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            cap = cv2.VideoCapture(temp_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if int(timestamp * self.fps) >= len(frames):
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
            cap.release()
            os.remove(temp_path)

        return chat_messages, StreamVideoContent(frames, self.fps)

    def stop(self):
        if self.chat:
            self.chat.terminate()
""".strip()

def format_cell_source(text):
    return [line + "\n" for line in text.splitlines()]

def create_notebook(filename, task_code, display_logic=False):
    cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!pip install -q yt-dlp pytchat opencv-python-headless numpy Pillow\n"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": format_cell_source(FETCHER_CODE)
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": format_cell_source(VIDEO_SOURCES_CODE)
        }
    ]
    
    # Process task code to replace imports and add display logic
    filtered_task_code = []
    lines = task_code.splitlines()
    for line in lines:
        if line.startswith("!pip") or "import stream_fetcher" in line or "import config" in line or "from stream_fetcher" in line or "from config" in line:
            continue
        if "sys.path.append" in line or "import sys" in line:
            continue
        filtered_task_code.append(line)
        
        # Add visual feedback for task1
        if display_logic and "recent_chat_list, video_content = fetcher.get_data_window(duration_sec=30)" in line:
            filtered_task_code.append("            from IPython.display import display")
            filtered_task_code.append("            from PIL import Image")
            filtered_task_code.append("            if video_content.frames:")
            filtered_task_code.append("                print('Rendering first frame of context:')")
            filtered_task_code.append("                display(Image.fromarray(video_content.frames[0]))")
            
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in filtered_task_code]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w') as f:
        json.dump(nb, f, indent=1)

# Task 1
with open('kaggle/task1.py', 'r') as f: t1 = f.read()
t1 = t1.split("# Benchmark Configuration")[1] # Keep only the bottom part
create_notebook('kaggle/task1.ipynb', "import textwrap\nimport kaggle_benchmarks as kbench\n" + t1, display_logic=True)

# Task 2
with open('kaggle/task2.py', 'r') as f: t2 = f.read()
create_notebook('kaggle/task2.ipynb', t2)

# Task 3
with open('kaggle/task3.py', 'r') as f: t3 = f.read()
create_notebook('kaggle/task3.ipynb', t3)
