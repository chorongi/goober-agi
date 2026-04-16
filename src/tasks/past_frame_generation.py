import time
import textwrap
import cv2
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
        fetcher = StreamFetcher(video_url, fps=1.0)
        try:
            fetcher.start()

            print("Providing 30s of initial context...")
            context_chat_list, video_content = fetcher.get_data_window(duration_sec=30)
            context_frames = video_content.frames
            context_chat_text = "\n".join(context_chat_list)

            print("Capturing withheld segment (10s)...")
            withheld_chat_list, withheld_video_content = fetcher.get_data_window(
                duration_sec=10
            )
            withheld_chat_text = "\n".join(withheld_chat_list)
            true_frames = withheld_video_content.frames

            # Resize both context and true frames to 224x224 to save tokens
            def resize_frames(frames):
                resized_list = []
                for f in frames:
                    resized = cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA)
                    resized_list.append(Image.fromarray(resized))
                return resized_list

            context_pil_frames = resize_frames(context_frames)
            true_pil_frames = resize_frames(true_frames)

            print("Generating Ground Truth Visual Description...")
            gt_prompt = textwrap.dedent(
                """
                You are an AI expert in visual analysis.
                Analyze the following 10 frames (10 seconds) of a video stream.

                Task: Provide a structured summary of the visual events.
                Format:
                [OVERALL SCENE]: (One sentence describing the environment and primary subject)
                [KEY EVENTS]: (Bullet points describing specific actions, screen changes, or visual movements occurring across the 10 seconds)

                Focus strictly on what is visually verifiable in the frames.
            """
            )
            gt_description = None
            for attempt in range(4):
                try:
                    gt_description = llm.prompt([gt_prompt, *true_pil_frames])
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

            if not gt_description:
                print("Failed to generate Ground Truth description.")
                continue

            prompt = textwrap.dedent(
                f"""
                You are an AI expert in visual reconstruction and social analysis.
                You have been watching a YouTube live stream. I have provided the last 30 seconds of video.

                Now, for the NEXT 10 seconds, the VIDEO IS WITHHELD. You only have the chat logs.

                --- PREVIOUS CHAT CONTEXT (0-30s) ---
                {context_chat_text}

                --- WITHHELD CHAT LOGS (30-40s) ---
                {withheld_chat_text}
                --- END CHAT ---

                Task: Based ON THE CHAT, reconstruct what happened visually during that withheld 10-second interval.

                Format (Match this EXACTLY):
                [OVERALL SCENE]: (One sentence describing the environment and primary subject)
                [KEY EVENTS]: (Bullet points describing specific actions, screen changes, or visual movements)

                Use the reactions in the chat to deduce the visual events.
            """
            )

            predicted_visuals = None
            for attempt in range(4):
                try:
                    predicted_visuals = llm.prompt([prompt, *context_pil_frames])
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
                "The [OVERALL SCENE] in the prediction accurately matches the ground truth environment.",
                "The [KEY EVENTS] described in the prediction align with the actions captured in the ground truth frames.",
                "The reconstruction shows a clear logical link between the chat reactions and the predicted visual events.",
            ]

            def judge_prompt_fn(criteria: list[str], response_text: str) -> str:
                formatted_criteria = "\n".join(f"- {c}" for c in criteria)
                return textwrap.dedent(
                    f"""
                    Evaluate the 'Predicted Reconstruction' against the 'Ground Truth Reconstruction'.
                    Both follow the [OVERALL SCENE] and [KEY EVENTS] format.

                    GROUND TRUTH RECONSTRUCTION (Actual Withheld Frames):
                    ```
                    {gt_description}
                    ```

                    PREDICTED RECONSTRUCTION (Generated from Chat):
                    ```
                    {response_text}
                    ```

                    CRITERIA:
                    {formatted_criteria}

                    Did the AI successfully 'see' the video through the eyes of the chat?
                """
                )

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
