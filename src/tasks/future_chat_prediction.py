import time
import textwrap
import cv2
import kaggle_benchmarks as kbench  # type: ignore
from PIL import Image

from ..stream_fetcher import StreamFetcher
from ..config import VIDEO_SOURCES


@kbench.task(name="future_chat_prediction", version=2)
def future_chat_prediction(llm) -> float:
    """
    Evaluates the AGI's ability to predict the next 10 seconds of chat
    using real-time data from 10 different YouTube live streams.
    """
    all_videos = [vid["url"] for category in VIDEO_SOURCES.values() for vid in category]
    total_score = 0.0
    valid_evals = 0

    for video_url in all_videos:
        print(f"\n--- Evaluating Video: {video_url} ---")
        fetcher = StreamFetcher(
            video_url, fps=1.0
        )  # 1.0 FPS for accurate visual context
        try:
            fetcher.start()

            # 1. Gather 30s of recent context
            print("Buffering 30s of history...")
            history_chat_list, history_video_content = fetcher.get_data_window(duration_sec=30)
            
            # Split history into Context and a small Example (last 3 messages) for few-shot
            context_chat_list = history_chat_list[:-3] if len(history_chat_list) > 3 else history_chat_list
            example_chat_list = history_chat_list[-3:] if len(history_chat_list) > 3 else []
            
            # Use 30 frames (30s at 1 FPS) and resize them to save context tokens
            raw_frames = history_video_content.frames
            resized_frames = []
            for frame in raw_frames:
                # Downsample to 224x224 to save significant context/tokens
                resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                resized_frames.append(resized)
            
            recent_chat_text = "\n".join(context_chat_list)
            example_chat_text = "\n".join(example_chat_list)

            # 2. Gather the ACTUAL next 10s of chat (Ground Truth)
            print("Gathering ground truth (next 10s)...")
            ground_truth_list, _ = fetcher.get_data_window(duration_sec=10)
            ground_truth_text = "\n".join(ground_truth_list)

            # 3. Prompt the AGI with dynamic few-shot example and better instructions
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

            # 4. Prompt LLM with retry logic
            predicted_chat = None
            for attempt in range(4):
                try:
                    predicted_chat = llm.prompt([prompt, *pil_frames])
                    break
                except Exception as e:
                    if attempt < 3 and (
                        "503" in str(e)
                        or "429" in str(e)
                        or "unavailable" in str(e).lower()
                    ):
                        print(f"LLM API unavailable ({e}), retrying in 10s...")
                        time.sleep(10)
                    else:
                        raise

            # 5. Evaluation Criteria for the Judge
            criteria = [
                "The predicted chat has high semantic similarity to the ground truth.",
                "The predicted chat captures the reaction flow and sequence.",
                "The predicted chat matches the likely sentiment of the stream participants.",
            ]

            def judge_prompt_fn(criteria: list[str], response_text: str) -> str:
                formatted_criteria = "\n".join(f"- {c}" for c in criteria)
                return textwrap.dedent(f"""
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
                """)

            # 6. Assess with Judge LLM
            assessment = None
            for attempt in range(4):
                try:
                    assessment = kbench.assertions.assess_response_with_judge(
                        criteria=criteria,
                        response_text=predicted_chat,
                        judge_llm=kbench.judge_llm,
                        prompt_fn=judge_prompt_fn,
                    )
                    break
                except Exception as e:
                    if attempt < 3 and (
                        "503" in str(e)
                        or "429" in str(e)
                        or "unavailable" in str(e).lower()
                    ):
                        print(f"Judge API unavailable ({e}), retrying in 10s...")
                        time.sleep(10)
                    else:
                        raise

            if assessment is not None:
                successes = 0
                for result in assessment.results:
                    kbench.assertions.assert_true(
                        result.passed,
                        expectation=f"[{video_url}] Criterion '{result.criterion}' failed. Reason: {result.reason}",
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
    print(f"\nFINAL SCORE ACROSS {valid_evals} STREAMS: {final_score:.2f}")
    return float(final_score)


if __name__ == "__main__":
    future_chat_prediction(kbench.llm)
