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
    Tests across all available stream pairs using a progressive 5s-30s polling loop (6 steps).
    """
    all_videos = [vid["url"] for category in VIDEO_SOURCES.values() for vid in category]
    total_score = 0.0
    valid_evals = 0

    for i in range(len(all_videos)):
        url_a = all_videos[i]
        url_b = all_videos[(i + 1) % len(all_videos)]

        print(f"\n--- Evaluating Switch Latency: {url_a} -> {url_b} ---")
        fetcher_a = StreamFetcher(url_a, fps=1.0)
        fetcher_b = StreamFetcher(url_b, fps=1.0)

        try:
            print("Gathering Stream A context (30s)...")
            fetcher_a.start()
            chat_a, video_a = fetcher_a.get_data_window(duration_sec=30)
            fetcher_a.stop()

            print("Starting Progressive Polling on Stream B (5s steps)...")
            fetcher_b.start()

            # Accumulated Stream B data
            chat_b_accumulated = []
            frames_b_accumulated = []

            adaptation_latency = 30  # Default to max penalty
            success_at_t = False

            def process_frames(frames):
                resized = []
                for f in frames:
                    r = cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA)
                    resized.append(Image.fromarray(r))
                return resized

            pil_frames_a = process_frames(video_a.frames)

            for t in range(5, 31, 5):
                print(f"  [t={t}s] Fetching new data and probing model...")

                # 1. Fetch exactly 5s of new data from Stream B
                new_chat_b, new_video_b = fetcher_b.get_data_window(duration_sec=5)
                chat_b_accumulated.extend(new_chat_b)
                frames_b_accumulated.extend(new_video_b.frames)

                # 2. Combine 30s of A + t seconds of B
                combined_chat_list = chat_a + chat_b_accumulated
                combined_chat_text = "\n".join(combined_chat_list)

                pil_frames_b = process_frames(frames_b_accumulated)
                all_pil_frames = pil_frames_a + pil_frames_b

                # 3. Predict the NEXT 10s
                # We gather ground truth for evaluation
                print(f"    -> Gathering GT for t={t}s probe...")
                gt_chat_list, _ = fetcher_b.get_data_window(duration_sec=10)
                gt_text = "\n".join(gt_chat_list)

                prompt = textwrap.dedent(
                    f"""
                    You are an AI expert in real-time stream analysis.
                    I have provided a sequence of video frames and the corresponding chat history.

                    --- CHAT HISTORY ---
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
                    "The model successfully recognized the stream change and did not hallucinate data from Stream A.",
                    "The prediction accurately reflects the genre, tone, and social dynamics of the SECOND stream (Stream B).",
                    "The prediction is semantically valid for the new ground truth chat of Stream B.",
                ]

                def judge_prompt_fn(criteria: list[str], response_text: str) -> str:
                    formatted_criteria = "\n".join(f"- {c}" for c in criteria)
                    return textwrap.dedent(
                        f"""
                        Evaluate if the model has successfully adapted to Stream B after a context switch.
                        The first 30s of context was Stream A. The last {t}s was Stream B.

                        GROUND TRUTH (STREAM B ACTUAL CHAT):
                        ```
                        {gt_text}
                        ```

                        MODEL PREDICTION:
                        ```
                        {response_text}
                        ```

                        CRITERIA:
                        {formatted_criteria}

                        Has the model effectively pivoted its world model to the new stream?
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
                    all_passed = all(result.passed for result in assessment.results)
                    if all_passed:
                        print(f"    SUCCESS! Model adapted at t={t} seconds.")
                        adaptation_latency = t
                        success_at_t = True
                        break
                    else:
                        print(f"    FAIL. Model still in contextual inertia at t={t}s.")
                else:
                    print(f"    ERROR. Judge failed at t={t}s.")

            # Score calculation: t=5s -> 100 pts, t=30s -> 0 pts
            # Formula: 100 * (1 - (latency - 5) / 25)
            if success_at_t:
                score = 100.0 * (1.0 - ((adaptation_latency - 5) / 25.0))
            else:
                score = 0.0

            total_score += score
            valid_evals += 1
            print(
                f"Adaptation Score for this switch: {score:.2f} (Latency: {adaptation_latency}s)"
            )

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
