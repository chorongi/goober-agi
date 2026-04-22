# Task 1: Social Foresight (Future Chat Prediction)

**Description**: Predict a representative sequence of chat messages that will appear in the live chat in the immediate future.

## Evaluation Methodology & Stabilization

To ensure a high-quality, statistically stable signal and a balanced level of difficulty, we implement several stabilization and noise-reduction techniques:

1. **Reduced Visual Noise (0.5 FPS)**:
   - Video frames are sampled at **0.5 FPS** (1 frame every 2 seconds).
   - This provides the model with clear temporal context of major visual shifts while minimizing token noise and prevent model distraction.

2. **Context Density & Temporal Guarantee**:
   - The benchmark buffers the stream until it meets **BOTH** of the following requirements:
     - At least **20 chat messages** have been captured.
     - At least **30 seconds** of video history has been captured.
   - This ensures the LLM always has a sufficient semantic and visual baseline.

3. **Representative Block Prediction**:
   - Instead of requiring an exact line-by-line prediction for a fixed time window, the model is asked to predict a **representative block of 5-10 messages**.
   - This allows the model to capture the *sentiment*, *vibe*, and *topics* of the crowd without being penalized for the chaotic timing of live chat.

4. **Statistical Smoothing (3-Trial Averaging)**:
   - The benchmark performs **3 consecutive prediction trials** per stream.
   - The final score is the average of these trials, reducing the impact of random anomalies.

## Evaluation Metrics (LLM-as-a-Judge)

The Judge LLM evaluates the prediction based on:
1. **Topic Anticipation**: Alignment with the actual content of the ground truth.
2. **Sentiment Alignment**: Matching the likely mood of the participants.
3. **Social Vibe**: Capturing the specific slang, reaction pace, and social dynamics.

### Implementation Note
By focusing on semantic and social relevance rather than temporal rigidity, Task 1 provides a discriminative signal of an AGI's capacity for social foresight.
