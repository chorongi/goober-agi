# !pip install -q yt-dlp pytchat opencv-python-headless numpy Pillow

import time
import os
import textwrap
import numpy as np
import kaggle_benchmarks as kbench  # type: ignore
from typing import List, Tuple
from PIL import Image

# Import the updated StreamFetcher
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
# from stream_fetcher import StreamFetcher

import os
import tempfile
import yt_dlp
import pytchat
import cv2


class StreamFetcher:
    def __init__(self, url: str, fps: float = 1.0):
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

    def get_data_window(self, duration_sec: int) -> Tuple[List[str], List[np.ndarray]]:
        """
        Fetches 'duration_sec' worth of chat messages and captures frames.
        Returns: (chat_messages, list_of_numpy_frames)
        """
        start_time = time.time()
        chat_messages = []
        frames = []

        # 1. Get the direct stream URL
        ydl_opts = {"format": "best[height<=360]", "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
            try:
                info = ydl.extract_info(self.url, download=False)
                if info is None or "url" not in info:
                    raise ValueError("Could not extract stream URL")
                stream_url = info["url"]  # type: ignore
            except Exception as e:
                print(f"Error extracting stream URL: {e}")
                return [], []

        # 2. Open stream with OpenCV
        cap = cv2.VideoCapture(str(stream_url))

        # 3. Poll chat and capture frames
        while time.time() - start_time < duration_sec:
            # Capture frame
            ret, frame = cap.read()
            if ret:
                # Capture at requested FPS
                if int((time.time() - start_time) * self.fps) > len(frames):
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)

            # Poll chat
            if self.chat and self.chat.is_alive():
                chat_data = self.chat.get()
                if hasattr(chat_data, "sync_items"):
                    for c in chat_data.sync_items():  # type: ignore
                        chat_messages.append(f"{c.author.name}: {c.message}")

            time.sleep(0.01)

        cap.release()

        # Final chat drain
        if self.chat and self.chat.is_alive():
            chat_data = self.chat.get()
            if hasattr(chat_data, "sync_items"):
                for c in chat_data.sync_items():  # type: ignore
                    chat_messages.append(f"{c.author.name}: {c.message}")

        return chat_messages, frames

    def stop(self):
        if self.chat:
            self.chat.terminate()


@kbench.task(name="future_chat_prediction", version=2)
def future_chat_prediction(llm, video_url: str) -> float:
    """
    Evaluates the AGI's ability to predict the next 10 seconds of chat
    using real-time data from the provided YouTube live stream.
    """
    fetcher = StreamFetcher(video_url, fps=1.0)  # 1.0 FPS for accurate visual context
    try:
        fetcher.start()

        # 1. Gather 30s of recent context
        print(f"Buffering 30s of history for {video_url}...")
        history_chat_list, raw_frames = fetcher.get_data_window(duration_sec=30)

        # Split history into Context and a small Example (last 3 messages) for few-shot
        context_chat_list = (
            history_chat_list[:-3] if len(history_chat_list) > 3 else history_chat_list
        )
        example_chat_list = history_chat_list[-3:] if len(history_chat_list) > 3 else []

        # Resize frames to 224x224 to save context tokens
        resized_frames = []
        for frame in raw_frames:
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            resized_frames.append(resized)

        recent_chat_text = "\n".join(context_chat_list)
        example_chat_text = "\n".join(example_chat_list)

        # 2. Prompt the AGI with dynamic few-shot example and better instructions
        prompt = textwrap.dedent(f"""
            You are an AI expert in analyzing online communities and social dynamics.
            You are observing a YouTube live stream. I have provided a sequence of video frames from the last 30 seconds of the stream.

            --- RECENT CHAT HISTORY ---
            {recent_chat_text}
            --- END CHAT HISTORY ---

            --- EXAMPLE OF RECENT MESSAGES (Format Reference) ---
            {example_chat_text}
            --- END EXAMPLE ---

            Task: Predict the most likely sequence and semantic flow of chat messages that will appear in the next 10 seconds.
            
            Guidelines:
            1. Capture the tone, slang, and emotional momentum of the crowd.
            2. Match the reaction pace (e.g., fast spam during action, slow chat during calm).
            3. Format your response exactly as: "username: message" (one per line).
            
            Predict the next 10 seconds of chat:
        """)

        # Convert numpy frames to PIL Images for the multimodal LLM
        pil_frames = [Image.fromarray(f) for f in resized_frames]

        # Hand the text prompt and frames to the LLM
        predicted_chat = llm.prompt([prompt, *pil_frames])

        # 3. Gather the ACTUAL next 10s of chat (Ground Truth)
        print("Gathering ground truth (next 10s)...")
        ground_truth_list, _ = fetcher.get_data_window(duration_sec=10)
        ground_truth_text = "\n".join(ground_truth_list)

        # 4. Evaluation Criteria for the Judge
        criteria = [
            "The predicted chat has high semantic similarity to the ground truth.",
            "The predicted chat captures the reaction flow and sequence.",
            "The predicted chat matches the likely sentiment of the stream participants.",
        ]

        def judge_prompt_fn(criteria: list[str], response_text: str) -> str:
            formatted_criteria = "\n".join(f"- {c}" for c in criteria)
            return textwrap.dedent(
                f"""
                Compare the 'Predicted Chat' against the 'Ground Truth Chat'.

                NOTE: Ignore the specific usernames (e.g. "User123:") and any "@" mentions.
                Focus strictly on the semantic content, reaction flow, and sentiment of the actual messages.

                GROUND TRUTH:
                ```
                {ground_truth_text}
                ```

                PREDICTED:
                ```
                {response_text}
                ```

                CRITERIA:
                {formatted_criteria}

                Evaluate if the prediction accurately anticipated the social state of the stream.
            """
            )

        # 5. Assess with Judge LLM
        assessment = kbench.assertions.assess_response_with_judge(
            criteria=criteria,
            response_text=predicted_chat,
            judge_llm=kbench.judge_llm,
            prompt_fn=judge_prompt_fn,
        )

        if assessment is None:
            kbench.assertions.assert_fail("Judge LLM assessment failed.")
            return 0.0

        successes = 0
        for result in assessment.results:
            kbench.assertions.assert_true(
                result.passed,
                expectation=f"Criterion '{result.criterion}' failed. Reason: {result.reason}",
            )
            if result.passed:
                successes += 1

        score = (successes / len(criteria)) * 100.0
        return float(score)

    finally:
        fetcher.stop()


future_chat_prediction(
    kbench.llm, video_url="https://www.youtube.com/watch?v=jfKfPfyJRdk"
)
