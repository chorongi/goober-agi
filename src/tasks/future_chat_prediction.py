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
    Evaluates the AGI's ability to predict the next 10 seconds of chat.
    Uses dynamic buffering to ensure context density and 3-trial averaging for stability.
    """
    all_videos = [vid["url"] for category in VIDEO_SOURCES.values() for vid in category]
    total_score = 0.0
    valid_evals = 0

    for video_url in all_videos:
        print(f"\n--- Evaluating Video: {video_url} ---")
        fetcher = StreamFetcher(video_url, fps=1.0)
        try:
            fetcher.start()

            # 1. Dynamic Buffering: Guarantee at least 20 messages AND at least 30s of video
            print("Buffering for context density (min 20 msgs AND 30s duration)...")
            history_chat_list = []
            history_video_content = None
            
            start_buffer_time = time.time()
            while (time.time() - start_buffer_time) < 120:
                # Get a small 10s chunk and accumulate
                msgs, video = fetcher.get_data_window(duration_sec=10)
                history_chat_list.extend(msgs)
                if history_video_content is None:
                    history_video_content = video
                else:
                    history_video_content.frames.extend(video.frames)
                
                elapsed = time.time() - start_buffer_time
                print(f"  Current buffer: {len(history_chat_list)} msgs, {elapsed:.0f}s duration...")
                
                # Stop if we meet BOTH density and duration requirements
                if len(history_chat_list) >= 20 and elapsed >= 30:
                    break

            # 2. Perform 3 trials for statistical smoothing
            stream_trial_scores = []
            for trial_idx in range(1, 4):
                print(f"  [Trial {trial_idx}/3] Probing model...")
                
                # Use the most recent 30s of gathered history for the prompt
                context_chat_list = history_chat_list[-20:] # Last 20 messages
                # Last 30 frames
                recent_frames = history_video_content.frames[-30:] if history_video_content else []
                
                # Split for few-shot (last 3 messages as example)
                prompt_chat_list = context_chat_list[:-3] if len(context_chat_list) > 3 else context_chat_list
                example_chat_list = context_chat_list[-3:] if len(context_chat_list) > 3 else []
                
                # Resize frames to 224x224
                resized_frames = []
                for frame in recent_frames:
                    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    resized_frames.append(Image.fromarray(resized))
                
                recent_chat_text = "\n".join(prompt_chat_list)
                example_chat_text = "\n".join(example_chat_list)

                # Gather the Ground Truth for THIS trial (next 10s)
                print(f"    -> Gathering Ground Truth for Trial {trial_idx}...")
                ground_truth_list, gt_video = fetcher.get_data_window(duration_sec=10)
                ground_truth_text = "\n".join(ground_truth_list)

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

                predicted_chat = None
                for attempt in range(4):
                    try:
                        predicted_chat = llm.prompt([prompt, *resized_frames])
                        break
                    except Exception as e:
                        if attempt < 3 and ("503" in str(e) or "429" in str(e) or "unavailable" in str(e).lower()):
                            print(f"LLM API unavailable ({e}), retrying in 10s...")
                            time.sleep(10)
                        else:
                            raise

                # 3. Evaluation Logic with Empty-GT Handling
                if not ground_truth_list:
                    # If GT is empty, model scores 100 if it predicted nothing/silence, 0 otherwise
                    is_pred_empty = len(str(predicted_chat).strip()) < 5 
                    trial_score = 100.0 if is_pred_empty else 0.0
                    print(f"    Trial {trial_idx} result: Ground Truth is EMPTY. Model Prediction Empty: {is_pred_empty}. Score: {trial_score}")
                else:
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
                            if attempt < 3 and ("503" in str(e) or "429" in str(e) or "unavailable" in str(e).lower()):
                                print(f"Judge API unavailable ({e}), retrying in 10s...")
                                time.sleep(10)
                            else:
                                raise

                    if assessment is not None:
                        successes = sum(1 for result in assessment.results if result.passed)
                        trial_score = (successes / len(criteria)) * 100.0
                    else:
                        trial_score = 0.0
                    
                stream_trial_scores.append(trial_score)
                print(f"    Trial {trial_idx} Score: {trial_score:.2f}")

                # Update history with the ground truth of the current trial for the next trial
                history_chat_list.extend(ground_truth_list)
                if history_video_content:
                    history_video_content.frames.extend(gt_video.frames)

            # Average score for the stream
            avg_stream_score = sum(stream_trial_scores) / len(stream_trial_scores)
            total_score += avg_stream_score
            valid_evals += 1
            print(f"FINAL STABILIZED SCORE for {video_url}: {avg_stream_score:.2f}")

        except Exception as e:
            print(f"Failed processing {video_url}: {e}")
        finally:
            fetcher.stop()

    if valid_evals == 0:
        kbench.assertions.assert_fail("All video evaluations failed.")
        return 0.0

    final_score = total_score / valid_evals
    print(f"\nFINAL STABILIZED SCORE ACROSS {valid_evals} STREAMS: {final_score:.2f}")
    return float(final_score)


if __name__ == "__main__":
    future_chat_prediction.run(kbench.llm)
