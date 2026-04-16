# Task 3: Stream-Switch Adaptation (Zero-Shot Latency)

**Description**: This task measures the "cognitive agility" of the AGI. It quantifies the time required for the model to discard an outdated world model and reach a functional level of predictive accuracy in a completely new, unseen environment (stream switch).

## Precise Execution Steps

1. **Baseline Phase**:
   - The AGI observes "Stream A" for a minimum of 5 minutes.
   - Calculate the average **Task 1 (Future Chat Prediction)** BERTScore over the final 60 seconds of this phase.
   - Let this baseline score be $S_{base}$.

2. **Threshold Calibration**:
   - Define the **Target Threshold** ($T_{target}$) as: 
     $T_{target} = S_{base} \times [THRESHOLD\_PERCENTAGE]$
   - *(Note: A common starting value is 0.85 or 85% of baseline performance).*

3. **The Hard Switch**:
   - At a timestamp $T_{switch}$, the input feed is instantly swapped from "Stream A" to "Stream B" (a different genre, game, or creator).
   - The AGI is not notified of the switch; it must detect the context shift through its own multimodal inputs.

4. **Recovery Monitoring**:
   - Continuously calculate the Task 1 BERTScore for every 10-second window following $T_{switch}$.
   - The recovery is considered "Complete" once the AGI's prediction score consistently stays above $T_{target}$ for three consecutive windows.

## Automated Ground Truth
The ground truth is the actual synchronized chat logs and video frames from "Stream B" immediately following the switch point.

## Evaluation Metric

1. **Zero-Shot Adaptation Latency (ZAL)**:
   - **Definition**: The time elapsed (in seconds) between $T_{switch}$ and the first second of the "Complete" recovery state.
   - **Formula**: $ZAL = T_{recovered} - T_{switch}$.
   - **Goal**: Minimize ZAL. A "General" intelligence should adapt to a new stream context (e.g., switching from FPS gameplay to a Cooking stream) in the time it takes for the first few chat reactions to appear.

## Implementation Note
To prevent "lucky" guesses, the benchmark requires the threshold to be maintained for a duration (3 windows) before the latency is recorded. This ensures the AGI has truly recalibrated its world model to the new stream's dynamics.
