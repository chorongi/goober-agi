# Task 2: Cognitive Reconstruction (Visual Inference)

**Description**: The AGI observes a continuous segment of video and chat. Then, for a 10-second interval, the video feed is withheld—it only receives the live chat logs. The AGI must reconstruct the visual state based *only* on the withheld chat logs.

## Evaluation Methodology & Stabilization

To ensure a high-quality, statistically stable signal of the model's visual reasoning, we implement several stabilization techniques:

1. **Context Density & Temporal Guarantee**:
   - The benchmark buffers the initial context until it meet **BOTH** of the following requirements:
     - At least **20 chat messages** have been captured.
     - At least **30 seconds** of video history has been captured.
   - This ensures the model has a sufficient understanding of the environment before being asked to infer hidden states.

2. **Ground Truth Reconstruction**:
   - To eliminate the ambiguity of comparing model text to raw video, the benchmark uses the multimodal LLM itself to generate a **Ground Truth Reconstruction** from the actual withheld frames.
   - Both the GT and the model's prediction are forced into a unified template:
     ```
     [OVERALL SCENE]: (Environment summary)
     [KEY EVENTS]: (Specific actions/movements)
     ```

3. **Statistical Smoothing (3-Trial Averaging)**:
   - For every stream, the benchmark performs **3 consecutive prediction trials**.
   - After each trial, the history is updated with the true frames and chat of the previous segment.
   - The final score for the stream is the average of these 3 trials, reducing the impact of momentary visual or social lulls.

## Evaluation Metrics (LLM-as-a-Judge)

The Judge LLM evaluates the `Predicted Reconstruction` against the `Ground Truth Reconstruction` based on:
1. **Scene Accuracy**: Alignment with the ground truth environment.
2. **Event Alignment**: Matching the specific actions captured in the frames.
3. **Logical Linking**: Demonstrating a clear link between chat reactions and visual events.

### Implementation Note
By using an apples-to-apples text comparison and averaging across multiple trials, Task 2 provides a robust measure of an AGI's ability to "see" through the eyes of a crowd.
