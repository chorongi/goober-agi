import os
import time
import tempfile
import base64
import yt_dlp
import pytchat
import cv2
import numpy as np
from typing import List, Tuple

try:
    from kaggle_benchmarks.content_types.videos import VideoContent
except ImportError:
    # Fallback/mock if kaggle_benchmarks isn't installed in the local path during testing
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
    """
    A custom VideoContent implementation that serves a pre-encoded mp4 
    (with audio and video) as a base64 Data URI, abiding by the 
    Kaggle Benchmarks API contract.
    """
    def __init__(self, frames: List[np.ndarray], fps: float, b64_data: str):
        self.frames = frames
        self.fps = fps
        self._b64_data = b64_data

    @property
    def url(self) -> str:
        # Returns a standard Data URI compliant with image_url spec
        return f"data:{self.mime_type};base64,{self._b64_data}"

    @property
    def mime_type(self) -> str:
        return "video/mp4"


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
        """Initializes the connection to the chat stream."""
        self.chat = pytchat.create(video_id=self.video_id)
        
    def get_data_window(self, duration_sec: int) -> Tuple[List[str], StreamVideoContent]:
        """
        Fetches 'duration_sec' worth of chat messages and captures an mp4 (audio+video).
        Returns: (chat_messages, StreamVideoContent_object)
        """
        import subprocess
        
        start_time = time.time()
        chat_messages = []
        frames = []
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
            temp_path = tf.name

        # 1. Download segment using yt-dlp's ffmpeg downloader
        # This properly handles YouTube's HLS cookies/headers and captures audio+video
        cmd = [
            'yt-dlp', 
            '-f', 'best[height<=360]', 
            '--downloader', 'ffmpeg', 
            '--downloader-args', f'ffmpeg:-t {duration_sec}', 
            '-o', temp_path,
            '--force-overwrites',
            '--quiet',
            self.url
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Poll chat while the video downloads
        while time.time() - start_time < duration_sec:
            if self.chat and self.chat.is_alive():
                for c in self.chat.get().sync_items():
                    chat_messages.append(f"{c.author.name}: {c.message}")
            time.sleep(0.1)

        # Wait for the download to finalize
        process.wait(timeout=15)
        
        # Final chat drain
        if self.chat and self.chat.is_alive():
            for c in self.chat.get().sync_items():
                chat_messages.append(f"{c.author.name}: {c.message}")
            
        b64_data = ""
        # 3. Extract frames for the metrics evaluator and read the mp4 data
        if os.path.exists(temp_path):
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

            # Read the raw mp4 file (which includes audio!)
            with open(temp_path, 'rb') as f:
                b64_data = base64.b64encode(f.read()).decode('utf-8')
                
            os.remove(temp_path)
            
        return chat_messages, StreamVideoContent(frames, self.fps, b64_data)

    def stop(self):
        if self.chat:
            self.chat.terminate()
