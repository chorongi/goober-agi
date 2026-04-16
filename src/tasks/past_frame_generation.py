import time
import textwrap
import kaggle_benchmarks as kbench  # type: ignore
from PIL import Image

from ..stream_fetcher import StreamFetcher
from ..config import VIDEO_SOURCES


@kbench.task(name="past_frame_generation", version=2)
def past_frame_generation(llm) -> float:
    """
    Evaluates the AGI's ability to reconstruct the visual state of a 10s segment
    given the previous 30s context and ONLY the chat logs from the withheld 10s interval.
    """
    all_videos = [vid["url"] for category in VIDEO_SOURCES.values() for vid in category]
    total_score = 0.0
    valid_evals = 0

    for video_url in all_videos:
        print(f"\n--- Evaluating Video: {video_url} ---")
        fetcher = StreamFetcher(video_url, fps=0.2)
        try:
            fetcher.start()

            print("Providing 30s of initial context...")
            context_chat_list, video_content = fetcher.get_data_window(duration_sec=30)
            context_frames = video_content.frames
            context_chat_text = "\n".join(context_chat_list)

            print("Capturing withheld segment (10s)...")
            withheld_chat_list, _ = fetcher.get_data_window(duration_sec=10)
            withheld_chat_text = "\n".join(withheld_chat_list)

            prompt = textwrap.dedent(f"""
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
            """)

            pil_frames = [Image.fromarray(f) for f in context_frames]

            predicted_visuals = None
            for attempt in range(4):
                try:
                    predicted_visuals = llm.prompt([prompt, *pil_frames])
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

            criteria = [
                "The reconstructed descriptions align with the events described in the withheld chat.",
                "The temporal sequence of events matches the chat's reaction timing.",
                "The visual descriptions are plausible for the context of this specific stream.",
            ]

            def judge_prompt_fn(criteria: list[str], response_text: str) -> str:
                formatted_criteria = "\n".join(f"- {c}" for c in criteria)
                return textwrap.dedent(f"""
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
                """)

            assessment = None
            for attempt in range(4):
                try:
                    assessment = kbench.assertions.assess_response_with_judge(
                        criteria=criteria,
                        response_text=predicted_visuals,
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
    past_frame_generation(kbench.llm)
