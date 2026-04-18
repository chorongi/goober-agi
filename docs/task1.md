# Task 1: Future Chat Prediction

**Description**: Predict the exact sequence of chat messages that will appear in the live chat during the interval $[T+1, T+10]$ seconds.

**Automated Ground Truth**: The actual chat messages that appear in the live stream between $T+1$ and $T+10$ seconds serve as the ground truth ($G$). 

## Evaluation Metrics & Stabilization

To effectively measure the quality of chat predictions and ensure high reproducibility, we implement two primary stabilization techniques:

1. **Context Density & Temporal Guarantee (Dynamic Buffering)**:
   - The benchmark buffers the stream until it meets **BOTH** of the following requirements (capped at 120 seconds):
     - At least **20 chat messages** have been captured.
     - At least **30 seconds** of video history has been captured.
   - This ensures the LLM always has a statistically significant visual and semantic baseline, preventing context truncation in high-velocity streams.

2. **Statistical Smoothing (3-Trial Averaging)**:
   - For every stream, the benchmark performs **3 consecutive prediction trials**.
   - The final score for the stream is the average of these 3 trials.
   - This averages out momentary anomalies (e.g., sudden bot spam or random silence) that could otherwise cause high variance.

3. **Empty-GT Handling**:
   - If the ground truth window is naturally empty (no one chatted):
     - If the model correctly predicts silence/emptiness: **Score = 100**.
     - If the model predicts an active chat: **Score = 0**.

## Scoring (LLM-as-a-Judge)

The prediction is evaluated using the `kaggle_benchmarks` judge against three criteria:
1. **Semantic Similarity**: High alignment with the ground truth intent.
2. **Reaction Flow**: Captured the correct sequence of reactions.
3. **Sentiment Match**: Matched the likely sentiment of the participants.

### Implementation Note
For the final benchmark score, the weighted average across all 10+ streams is calculated, providing a stable and reliable metric of social foresight.
