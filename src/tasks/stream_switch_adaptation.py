import time
import textwrap
import cv2
import kaggle_benchmarks as kbench  # type: ignore
from PIL import Image

from ..stream_fetcher import StreamFetcher
from ..config import VIDEO_SOURCES


@kbench.task(name="stream_switch_adaptation", version=2)
def stream_switch_adaptation(llm) -> float:
    """
    Evaluates Zero-Shot Adaptation Latency by switching from Stream A to Stream B
    and measuring the quality of the first few predictions on the new stream.
    Tests across all available stream pairs.
    """
    all_videos = [vid["url"] for category in VIDEO_SOURCES.values() for vid in category]
    total_score = 0.0
    valid_evals = 0

    for i in range(len(all_videos)):
        url_a = all_videos[i]
        url_b = all_videos[(i + 1) % len(all_videos)]

        print(f"\n--- Evaluating Switch: {url_a} -> {url_b} ---")
        fetcher_a = StreamFetcher(url_a, fps=1.0)
        fetcher_b = StreamFetcher(url_b, fps=1.0)

        try:
            print("Gathering Stream A context (30s)...")
            fetcher_a.start()
            chat_a, video_a = fetcher_a.get_data_window(duration_sec=30)
            fetcher_a.stop()

            print("Gathering Stream B context (30s glance)...")
            fetcher_b.start()
            chat_b, video_b = fetcher_b.get_data_window(duration_sec=30)

            # Combine context: 30s of A + 30s of B
            combined_chat_list = chat_a + chat_b
            combined_chat_text = "\n".join(combined_chat_list)

            # Combine frames and resize to 224x224
            def process_frames(frames):
                resized = []
                for f in frames:
                    r = cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA)
                    resized.append(Image.fromarray(r))
                return resized

            all_pil_frames = process_frames(video_a.frames + video_b.frames)

            print("Gathering ground truth for Stream B (next 10s)...")
            gt_chat_list, _ = fetcher_b.get_data_window(duration_sec=10)
            gt_text = "\n".join(gt_chat_list)

            # Note: No explicit "CONTEXT SWITCH" notification in the prompt text
            prompt = textwrap.dedent(
                f"""
                You are an AI expert in real-time stream analysis.
                I have provided a 60-second sequence of video frames and the corresponding chat history.

                --- CHAT HISTORY (60 Seconds) ---
                {combined_chat_text}
                --- END CHAT ---

                Task: Predict the chat messages for the NEXT 10 seconds.
                Format: "username: message" (one per line).

                Important: Ensure your prediction is grounded in the CURRENT state of the stream you are seeing at the END of the provided context.
            """
            )

            zero_shot_prediction = None
            for attempt in range(4):
                try:
                    zero_shot_prediction = llm.prompt([prompt, *all_pil_frames])
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
                "The model successfully recognized the stream change and did not hallucinate data from the first half of the context (Stream A).",
                "The prediction accurately reflects the genre, tone, and social dynamics of the SECOND stream (Stream B).",
                "The prediction is semantically valid for the new ground truth chat of Stream B.",
            ]

            def judge_prompt_fn(criteria: list[str], response_text: str) -> str:
                formatted_criteria = "\n".join(f"- {c}" for c in criteria)
                return textwrap.dedent(
                    f"""
                    You are a STRICT and UNYIELDING evaluator testing an AI's Zero-Shot Adaptation capabilities.
                    The AI was switched from a previous context to this new stream with only a 5-second glance.

                    GROUND TRUTH (NEW STREAM'S ACTUAL CHAT):
                    ```
                    {gt_text}
                    ```

                    MODEL PREDICTION (ZERO-SHOT):
                    ```
                    {response_text}
                    ```

                    CRITERIA TO EVALUATE:
                    {formatted_criteria}

                    STRICT GRADING RULES:
                    1. DO NOT pass the prediction if it consists only of generic, low-effort responses (e.g., just "lol", "gg", "hi") UNLESS the Ground Truth is also exclusively those generic words.
                    2. The prediction MUST contain specific semantic links to the unique context of the new stream (e.g., mentioning specific game mechanics, the streamer's actions, or specific topics from the ground truth).
                    3. If the prediction carries over distinct vocabulary or topics that are completely absent from the new ground truth, fail the 'recognized context switch' criterion immediately.

                    Evaluate rigorously. Did the model truly adapt to the new, specific stream context?
                """
                )

            assessment = None
            for attempt in range(4):
                try:
                    assessment = kbench.assertions.assess_response_with_judge(
                        criteria=criteria,
                        response_text=zero_shot_prediction,
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
                        expectation=f"[{url_b}] Criterion '{result.criterion}' failed. Reason: {result.reason}",
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
    print(f"\nFINAL SCORE ACROSS {valid_evals} SWITCHES: {final_score:.2f}")
    return float(final_score)


if __name__ == "__main__":
    stream_switch_adaptation(kbench.llm)
