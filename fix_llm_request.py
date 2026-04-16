import json

TASK1_PY = """
import time
import os
import textwrap
import numpy as np
import kaggle_benchmarks as kbench
from typing import List, Tuple
from PIL import Image

# Task Parameters
TASK1_WINDOW_SEC = 10

@kbench.task(name="future_chat_prediction", version=2)
def future_chat_prediction(llm) -> float:
    \"\"\"
    Evaluates the AGI's ability to predict the next 10 seconds of chat
    using real-time data from 10 different YouTube live streams.
    \"\"\"
    all_videos = [vid['url'] for category in VIDEO_SOURCES.values() for vid in category]
    total_score = 0.0
    valid_evals = 0

    for video_url in all_videos:
        print(f"\\n--- Evaluating Video: {video_url} ---")
        fetcher = StreamFetcher(video_url, fps=0.2) # Capturing 1 frame every 5s for context
        try:
            fetcher.start()

            # 1. Gather 30s of recent context
            print("Buffering 30s of history...")
            recent_chat_list, frames = fetcher.get_data_window(duration_sec=30)
            
            from IPython.display import display
            if frames:
                print('Rendering first frame of context:')
                display(Image.fromarray(frames[0]))
                
            recent_chat_text = "\\n".join(recent_chat_list)

            # 2. Gather the ACTUAL next 10s of chat (Ground Truth) BEFORE prompting
            print("Gathering ground truth (next 10s)...")
            ground_truth_list, _ = fetcher.get_data_window(duration_sec=10)
            ground_truth_text = "\\n".join(ground_truth_list)

            # 3. Prompt the AGI with both chat and visual context
            prompt = textwrap.dedent(f\"\"\"
                You are an AI expert in analyzing online communities.
                You are observing a YouTube live stream. I have provided a video clip of the last 30 seconds of the stream.

                --- RECENT CHAT HISTORY (Last 30 Seconds) ---
                {recent_chat_text}
                --- END CHAT HISTORY ---

                Task: Predict the EXACT sequence of chat messages that will appear in the next 10 seconds.
                Format: "username: message" (one per line).
            \"\"\")

            # Convert numpy frames to PIL Images for the multimodal LLM
            pil_frames = [Image.fromarray(f) for f in frames]

            # 4. Prompt LLM with retry logic
            predicted_chat = None
            for attempt in range(4):
                try:
                    predicted_chat = llm.prompt([prompt, *pil_frames])
                    break
                except Exception as e:
                    if attempt < 3 and ('503' in str(e) or '429' in str(e) or 'unavailable' in str(e).lower()):
                        print(f"LLM API unavailable ({e}), retrying in 10s...")
                        time.sleep(10)
                    else:
                        raise

            # 5. Evaluation Criteria for the Judge
            criteria = [
                "The predicted chat has high semantic similarity to the ground truth.",
                "The predicted chat captures the reaction flow and sequence.",
                "The predicted chat matches the likely sentiment of the stream participants."
            ]

            def judge_prompt_fn(criteria: list[str], response_text: str) -> str:
                formatted_criteria = "\\n".join(f"- {c}" for c in criteria)
                return textwrap.dedent(f\"\"\"
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
                \"\"\")

            # 6. Assess with Judge LLM
            assessment = None
            for attempt in range(4):
                try:
                    assessment = kbench.assertions.assess_response_with_judge(
                        criteria=criteria,
                        response_text=predicted_chat,
                        judge_llm=kbench.judge_llm,
                        prompt_fn=judge_prompt_fn
                    )
                    break
                except Exception as e:
                    if attempt < 3 and ('503' in str(e) or '429' in str(e) or 'unavailable' in str(e).lower()):
                        print(f"Judge API unavailable ({e}), retrying in 10s...")
                        time.sleep(10)
                    else:
                        raise

            if assessment is not None:
                successes = 0
                for result in assessment.results:
                    kbench.assertions.assert_true(
                        result.passed,
                        expectation=f"[{video_url}] Criterion '{result.criterion}' failed. Reason: {result.reason}"
                    )
                    if result.passed:
                        successes += 1

                score = (successes / len(criteria)) * 100.0
                total_score += score
                valid_evals += 1
                print(f"Score for this stream: {score:.2f}")
            else:
                print("Judge assessment failed for this stream.")

        except Exception as e:
            print(f"Failed processing {video_url}: {e}")
        finally:
            fetcher.stop()

    if valid_evals == 0:
        kbench.assertions.assert_fail("All 10 video evaluations failed.")
        return 0.0

    final_score = total_score / valid_evals
    print(f"\\nFINAL SCORE ACROSS {valid_evals} STREAMS: {final_score:.2f}")
    return float(final_score)

if __name__ == "__main__":
    future_chat_prediction.run(kbench.llm)
"""

TASK2_PY = """
import time
import os
import textwrap
import numpy as np
import kaggle_benchmarks as kbench
from typing import List, Tuple
from PIL import Image

@kbench.task(name="past_frame_generation", version=2)
def past_frame_generation(llm) -> float:
    \"\"\"
    Evaluates the AGI's ability to reconstruct the visual state of a 10s segment
    given the previous 30s context and ONLY the chat logs from the withheld 10s interval.
    \"\"\"
    all_videos = [vid['url'] for category in VIDEO_SOURCES.values() for vid in category]
    total_score = 0.0
    valid_evals = 0

    for video_url in all_videos:
        print(f"\\n--- Evaluating Video: {video_url} ---")
        fetcher = StreamFetcher(video_url, fps=0.2)
        try:
            fetcher.start()
            
            print("Providing 30s of initial context...")
            context_chat_list, context_frames = fetcher.get_data_window(duration_sec=30)
            context_chat_text = "\\n".join(context_chat_list)

            print("Capturing withheld segment (10s)...")
            withheld_chat_list, ground_truth_frames = fetcher.get_data_window(duration_sec=10)
            withheld_chat_text = "\\n".join(withheld_chat_list)

            prompt = textwrap.dedent(f\"\"\"
                You are an AI expert in visual reconstruction.
                You have been watching a YouTube live stream. I have provided the last 30 seconds of video.
                
                Now, for the NEXT 10 seconds, the VIDEO IS WITHHELD. You only have the chat logs.

                --- PREVIOUS CHAT CONTEXT (0-30s) ---
                {context_chat_text}

                --- WITHHELD CHAT LOGS (30-40s) ---
                {withheld_chat_text}
                --- END CHAT ---

                Task: Reconstruct what happened visually during that withheld 10-second interval. 
                Describe exactly 10 frames (one for each second). 
                Include details on the environment, screen changes, and the actions occurring.
            \"\"\")
            
            pil_frames = [Image.fromarray(f) for f in context_frames]
            
            predicted_visuals = None
            for attempt in range(4):
                try:
                    predicted_visuals = llm.prompt([prompt, *pil_frames])
                    break
                except Exception as e:
                    if attempt < 3 and ('503' in str(e) or '429' in str(e) or 'unavailable' in str(e).lower()):
                        print(f"LLM API unavailable ({e}), retrying in 10s...")
                        time.sleep(10)
                    else:
                        raise

            criteria = [
                "The reconstructed descriptions align with the events described in the withheld chat.",
                "The temporal sequence of events matches the chat's reaction timing.",
                "The visual descriptions are plausible for the context of this specific stream."
            ]

            def judge_prompt_fn(criteria: list[str], response_text: str) -> str:
                formatted_criteria = "\\n".join(f"- {c}" for c in criteria)
                return textwrap.dedent(f\"\"\"
                    The 'Predicted Reconstruction' was generated based on context and withheld chat.
                    Evaluate it against the 'Withheld Chat Context'.
                    
                    WITHHELD CHAT CONTEXT:
                    ```
                    {withheld_chat_text}
                    ```

                    PREDICTED RECONSTRUCTION:
                    ```
                    {response_text}
                    ```

                    CRITERIA:
                    {formatted_criteria}
                    
                    Did the AI successfully 'see' the video through the eyes of the chat?
                \"\"\")

            assessment = None
            for attempt in range(4):
                try:
                    assessment = kbench.assertions.assess_response_with_judge(
                        criteria=criteria,
                        response_text=predicted_visuals,
                        judge_llm=kbench.judge_llm,
                        prompt_fn=judge_prompt_fn
                    )
                    break
                except Exception as e:
                    if attempt < 3 and ('503' in str(e) or '429' in str(e) or 'unavailable' in str(e).lower()):
                        print(f"Judge API unavailable ({e}), retrying in 10s...")
                        time.sleep(10)
                    else:
                        raise

            if assessment is not None:
                successes = 0
                for result in assessment.results:
                    kbench.assertions.assert_true(
                        result.passed,
                        expectation=f"[{video_url}] Criterion '{result.criterion}' failed. Reason: {result.reason}"
                    )
                    if result.passed:
                        successes += 1

                score = (successes / len(criteria)) * 100.0
                total_score += score
                valid_evals += 1
                print(f"Score for this stream: {score:.2f}")
            else:
                print("Judge assessment failed for this stream.")

        except Exception as e:
            print(f"Failed processing {video_url}: {e}")
        finally:
            fetcher.stop()

    if valid_evals == 0:
        kbench.assertions.assert_fail("All 10 video evaluations failed.")
        return 0.0

    final_score = total_score / valid_evals
    print(f"\\nFINAL SCORE ACROSS {valid_evals} STREAMS: {final_score:.2f}")
    return float(final_score)

if __name__ == "__main__":
    past_frame_generation.run(kbench.llm)
"""


TASK3_PY = """
import time
import os
import textwrap
import numpy as np
import kaggle_benchmarks as kbench
from typing import List, Tuple
from PIL import Image

@kbench.task(name="stream_switch_adaptation", version=2)
def stream_switch_adaptation(llm) -> float:
    \"\"\"
    Evaluates Zero-Shot Adaptation Latency by switching from Stream A to Stream B
    and measuring the quality of the first few predictions on the new stream.
    Tests across all available stream pairs.
    \"\"\"
    all_videos = [vid['url'] for category in VIDEO_SOURCES.values() for vid in category]
    total_score = 0.0
    valid_evals = 0

    for i in range(len(all_videos)):
        url_a = all_videos[i]
        url_b = all_videos[(i + 1) % len(all_videos)]
        
        print(f"\\n--- Evaluating Switch: {url_a} -> {url_b} ---")
        fetcher_a = StreamFetcher(url_a, fps=0.2)
        fetcher_b = StreamFetcher(url_b, fps=0.2)
        
        try:
            print(f"Warming up on Stream A...")
            fetcher_a.start()
            _, _ = fetcher_a.get_data_window(duration_sec=30)
            fetcher_a.stop()

            print(f"HARD SWITCH to Stream B...")
            fetcher_b.start()
            
            print("Capturing 5s glance on Stream B...")
            glance_chat, glance_frames = fetcher_b.get_data_window(duration_sec=5)
            glance_text = "\\n".join(glance_chat)

            print("Gathering ground truth for Stream B (next 10s)...")
            gt_chat_list, _ = fetcher_b.get_data_window(duration_sec=10)
            gt_text = "\\n".join(gt_chat_list)

            prompt = textwrap.dedent(f\"\"\"
                CONTEXT SWITCH: You have just been switched to a NEW stream.
                I have provided a 5-second video clip of this new stream.
                
                --- FIRST 5 SECONDS OF NEW CHAT ---
                {glance_text}
                --- END CHAT ---

                Task: Predict the chat messages for the NEXT 10 seconds of this new stream.
                Format: "username: message" (one per line).
            \"\"\")
            
            pil_frames = [Image.fromarray(f) for f in glance_frames]
            
            zero_shot_prediction = None
            for attempt in range(4):
                try:
                    zero_shot_prediction = llm.prompt([prompt, *pil_frames])
                    break
                except Exception as e:
                    if attempt < 3 and ('503' in str(e) or '429' in str(e) or 'unavailable' in str(e).lower()):
                        print(f"LLM API unavailable ({e}), retrying in 10s...")
                        time.sleep(10)
                    else:
                        raise

            criteria = [
                "The model successfully recognized the context switch (did not hallucinate Stream A data).",
                "The zero-shot prediction shows understanding of the new stream's genre and social dynamics.",
                "The prediction is semantically valid for the new ground truth."
            ]

            def judge_prompt_fn(criteria: list[str], response_text: str) -> str:
                formatted_criteria = "\\n".join(f"- {c}" for c in criteria)
                return textwrap.dedent(f\"\"\"
                    Evaluating Zero-Shot Adaptation.
                    Model was switched from a previous context to this new stream.
                    
                    GROUND TRUTH (NEW STREAM):
                    ```
                    {gt_text}
                    ```

                    MODEL PREDICTION (ZERO-SHOT):
                    ```
                    {response_text}
                    ```

                    CRITERIA:
                    {formatted_criteria}
                    
                    Did the model adapt instantly to the new stream context?
                \"\"\")

            assessment = None
            for attempt in range(4):
                try:
                    assessment = kbench.assertions.assess_response_with_judge(
                        criteria=criteria,
                        response_text=zero_shot_prediction,
                        judge_llm=kbench.judge_llm,
                        prompt_fn=judge_prompt_fn
                    )
                    break
                except Exception as e:
                    if attempt < 3 and ('503' in str(e) or '429' in str(e) or 'unavailable' in str(e).lower()):
                        print(f"Judge API unavailable ({e}), retrying in 10s...")
                        time.sleep(10)
                    else:
                        raise

            if assessment is not None:
                successes = 0
                for result in assessment.results:
                    kbench.assertions.assert_true(
                        result.passed,
                        expectation=f"[{url_b}] Criterion '{result.criterion}' failed. Reason: {result.reason}"
                    )
                    if result.passed:
                        successes += 1

                score = (successes / len(criteria)) * 100.0
                total_score += score
                valid_evals += 1
                print(f"Score for this switch: {score:.2f}")
            else:
                print("Judge assessment failed for this switch.")

        except Exception as e:
            print(f"Failed processing switch {url_a} -> {url_b}: {e}")
        finally:
            fetcher_a.stop()
            fetcher_b.stop()

    if valid_evals == 0:
        kbench.assertions.assert_fail("All stream-switch evaluations failed.")
        return 0.0

    final_score = total_score / valid_evals
    print(f"\\nFINAL SCORE ACROSS {valid_evals} SWITCHES: {final_score:.2f}")
    return float(final_score)

if __name__ == "__main__":
    stream_switch_adaptation.run(kbench.llm)
"""

FETCHER_CODE = """
import os
import time
import tempfile
import yt_dlp
import pytchat
import cv2
import numpy as np
from typing import List, Tuple

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
        \"\"\"Initializes the connection to the chat stream.\"\"\"
        self.chat = pytchat.create(video_id=self.video_id)

    def get_data_window(self, duration_sec: int) -> Tuple[List[str], List[np.ndarray]]:
        \"\"\"
        Fetches 'duration_sec' worth of chat messages and captures frames.
        Returns: (chat_messages, list_of_numpy_frames)
        \"\"\"
        start_time = time.time()
        chat_messages = []
        frames = []

        # 1. Get the direct stream URL
        ydl_opts = {"format": "best[height<=360]", "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(self.url, download=False)
                stream_url = info["url"]
            except Exception as e:
                print(f"Error extracting stream URL: {e}")
                return [], []

        # 2. Open stream with OpenCV
        cap = cv2.VideoCapture(stream_url)

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
                for c in self.chat.get().sync_items():
                    chat_messages.append(f"{c.author.name}: {c.message}")

            time.sleep(0.01)

        cap.release()

        # Final chat drain
        if self.chat and self.chat.is_alive():
            for c in self.chat.get().sync_items():
                chat_messages.append(f"{c.author.name}: {c.message}")

        return chat_messages, frames

    def stop(self):
        if self.chat:
            self.chat.terminate()
"""

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
"""

def string_to_source_list(code_str):
    lines = code_str.strip().split('\n')
    return [line + "\\n" for line in lines[:-1]] + [lines[-1]] if lines else []

def create_notebook(filename, task_code):
    cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": string_to_source_list("!pip install -q yt-dlp pytchat opencv-python-headless numpy Pillow")
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": string_to_source_list(FETCHER_CODE)
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": string_to_source_list(VIDEO_SOURCES_CODE)
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": string_to_source_list(task_code)
        }
    ]
    
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

create_notebook('kaggle/task1.ipynb', TASK1_PY)
create_notebook('kaggle/task2.ipynb', TASK2_PY)
create_notebook('kaggle/task3.ipynb', TASK3_PY)
