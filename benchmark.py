import numpy as np
from typing import List

from src.config import VIDEO_SOURCES, TASK1_WINDOW_SEC, TASK2_WINDOW_SEC, WEIGHTS
from src.stream_fetcher import StreamFetcher
from src.metrics import Evaluator


class MockLLMInterface:
    """Mock interface representing the AGI/LLM under test."""

    def predict_future_chat(
        self, history_frames: List[np.ndarray], history_chat: List[str]
    ) -> List[str]:
        # Mock prediction: just echoing some history or generic responses
        return ["lol", "omg", "did you see that", "gg"]

    def generate_past_frames(self, history_chat: List[str]) -> List[np.ndarray]:
        # Mock prediction: returning 10 random noise frames (or black frames)
        # In reality, this would be a Diffusion Model generating 10 frames
        return [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(10)]


class AGIBenchmark:
    def __init__(self):
        self.evaluator = Evaluator()
        self.llm = MockLLMInterface()
        self.results = {"task1": [], "task2": [], "task3": []}

    def run_task1(self, fetcher: StreamFetcher) -> float:
        """Executes Task 1: Future Chat Prediction"""
        print("    -> Running Task 1...")
        # Get 60s of context
        history_chat, history_video = fetcher.get_data_window(60)

        # LLM Predicts next 10s of chat
        predicted_chat = self.llm.predict_future_chat(
            history_video.frames, history_chat
        )

        # Get actual 10s of chat (Ground Truth)
        true_chat, _ = fetcher.get_data_window(TASK1_WINDOW_SEC)

        score = self.evaluator.evaluate_task1(predicted_chat, true_chat)
        self.results["task1"].append(score)
        return score

    def run_task2(self, fetcher: StreamFetcher) -> float:
        """Executes Task 2: Past Frame Generation"""
        print("    -> Running Task 2...")
        # Get 10s of chat (withheld video)
        chat_context, true_video = fetcher.get_data_window(TASK2_WINDOW_SEC)

        # LLM Reconstructs the 10 frames
        predicted_frames = self.llm.generate_past_frames(chat_context)

        score = self.evaluator.evaluate_task2(
            predicted_frames, true_video.frames, chat_context
        )
        self.results["task2"].append(score)
        return score

    def run_task3(self, fetcher_a: StreamFetcher, fetcher_b: StreamFetcher) -> float:
        """Executes Task 3: Stream-Switch Adaptation (Zero-Shot Latency)"""
        print("    -> Running Task 3 (Stream Switch)...")
        # 1. Baseline on Stream A
        baseline_scores = []
        for _ in range(3):  # 3 windows to establish baseline
            history_chat, _ = fetcher_a.get_data_window(10)
            true_chat, _ = fetcher_a.get_data_window(10)
            pred = self.llm.predict_future_chat([], history_chat)
            baseline_scores.append(self.evaluator.evaluate_task1(pred, true_chat))

        baseline_score = np.mean(baseline_scores) if baseline_scores else 50.0
        target_threshold = baseline_score * 0.85  # 85% of baseline

        # 2. Hard Switch to Stream B
        recovery_windows = 0
        zal_seconds = 60  # Max timeout

        for i in range(6):  # Max 60 seconds (6 * 10s windows)
            history_chat, _ = fetcher_b.get_data_window(10)
            true_chat, _ = fetcher_b.get_data_window(10)
            pred = self.llm.predict_future_chat([], history_chat)
            score = self.evaluator.evaluate_task1(pred, true_chat)

            if score >= target_threshold:
                recovery_windows += 1
                if recovery_windows >= 3:  # 3 consecutive windows above threshold
                    zal_seconds = (i - 2) * 10
                    break
            else:
                recovery_windows = 0

        # Convert ZAL (lower is better, 0-60s) to a 0-100 score
        # 0s = 100 pts, 60s = 0 pts
        zal_score = max(0.0, 100.0 - (zal_seconds / 60.0 * 100.0))
        self.results["task3"].append(zal_score)
        return zal_score

    def run_full_suite(self):
        print("Starting MM Goober Benchmark Suite...")
        # Flatten video list
        all_videos = [vid for category in VIDEO_SOURCES.values() for vid in category]

        for i, video in enumerate(all_videos):
            print(f"\n[{i + 1}/{len(all_videos)}] Evaluating on: {video['name']}")
            try:
                fetcher = StreamFetcher(video["url"])
                fetcher.start()

                # Warmup buffer
                print("    -> Buffering stream...")
                fetcher.get_data_window(5)

                self.run_task1(fetcher)
                self.run_task2(fetcher)

                # For Task 3, we switch to the next video in the list
                next_video = all_videos[(i + 1) % len(all_videos)]
                fetcher_b = StreamFetcher(next_video["url"])
                fetcher_b.start()
                self.run_task3(fetcher, fetcher_b)

                fetcher.stop()
                fetcher_b.stop()

            except Exception as e:
                print(f"    -> Error processing {video['name']}: {e}")

        # Calculate Final Scores
        avg_t1 = np.mean(self.results["task1"]) if self.results["task1"] else 0
        avg_t2 = np.mean(self.results["task2"]) if self.results["task2"] else 0
        avg_t3 = np.mean(self.results["task3"]) if self.results["task3"] else 0

        final_score = (
            (avg_t1 * WEIGHTS["task1"])
            + (avg_t2 * WEIGHTS["task2"])
            + (avg_t3 * WEIGHTS["task3"])
        )

        print("\n" + "=" * 40)
        print("🏆 MM GOOBER BENCHMARK RESULTS 🏆")
        print("=" * 40)
        print(f"Task 1 (Future Chat):     {avg_t1:.2f} / 100")
        print(f"Task 2 (Past Frames):     {avg_t2:.2f} / 100")
        print(f"Task 3 (ZAL Switch):      {avg_t3:.2f} / 100")
        print("-" * 40)
        print(f"🌟 FINAL AGI SCORE:       {final_score:.2f} / 100")
        print("=" * 40)


if __name__ == "__main__":
    benchmark = AGIBenchmark()
    benchmark.run_full_suite()
