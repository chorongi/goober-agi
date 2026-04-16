import time
import os
import textwrap
import numpy as np
import kaggle_benchmarks as kbench
from typing import List, Tuple
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
    all_videos = [vid['url'] for category in VIDEO_SOURCES.values() for vid in category]
    total_score = 0.0
    valid_evals = 0

    for i in range(len(all_videos)):
        url_a = all_videos[i]
        url_b = all_videos[(i + 1) % len(all_videos)]
        
        print(f"\n--- Evaluating Switch: {url_a} -> {url_b} ---")
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
            glance_chat, video_content = fetcher_b.get_data_window(duration_sec=5)
            glance_frames = video_content.frames
            glance_text = "\n".join(glance_chat)

            print("Gathering ground truth for Stream B (next 10s)...")
            gt_chat_list, _ = fetcher_b.get_data_window(duration_sec=10)
            gt_text = "\n".join(gt_chat_list)

            prompt = textwrap.dedent(f"""
                CONTEXT SWITCH: You have just been switched to a NEW stream.
                I have provided a 5-second video clip of this new stream.
                
                --- FIRST 5 SECONDS OF NEW CHAT ---
                {glance_text}
                --- END CHAT ---

                Task: Predict the chat messages for the NEXT 10 seconds of this new stream.
                Format: "username: message" (one per line).
            """)
            
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
                formatted_criteria = "\n".join(f"- {c}" for c in criteria)
                return textwrap.dedent(f"""
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
                """)

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
    print(f"\nFINAL SCORE ACROSS {valid_evals} SWITCHES: {final_score:.2f}")
    return float(final_score)

if __name__ == "__main__":
    stream_switch_adaptation.run(kbench.llm)
