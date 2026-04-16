import time
import textwrap
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
            video_url, fps=0.2
        )  # Capturing 1 frame every 5s for context
        try:
            fetcher.start()

            # 1. Gather 30s of recent context
            print("Buffering 30s of history...")
            recent_chat_list, video_content = fetcher.get_data_window(duration_sec=30)
            frames = video_content.frames

            recent_chat_text = "\n".join(recent_chat_list)

            # 2. Gather the ACTUAL next 10s of chat (Ground Truth) BEFORE prompting
            print("Gathering ground truth (next 10s)...")
            ground_truth_list, _ = fetcher.get_data_window(duration_sec=10)
            ground_truth_text = "\n".join(ground_truth_list)

            # 3. Prompt the AGI with both chat and visual context
            prompt = textwrap.dedent(f"""
                You are an AI expert in analyzing online communities.
                You are observing a YouTube live stream. I have provided a video clip of the last 30 seconds of the stream.

                --- RECENT CHAT HISTORY (Last 30 Seconds) ---
                {recent_chat_text}
                --- END CHAT HISTORY ---

                Task: Predict the EXACT sequence of chat messages that will appear in the next 10 seconds.
                Format: "username: message" (one per line).
            """)

            # Convert numpy frames to PIL Images for the multimodal LLM
            pil_frames = [Image.fromarray(f) for f in frames]

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
