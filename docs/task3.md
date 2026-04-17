# Task 3: Stream-Switch Adaptation (Context Agility)

**Description**: This task measures the "cognitive agility" of the AGI. It quantifies the model's ability to perform a **Context Switch**: discarding an outdated world model and instantly reaching a functional level of predictive accuracy in a completely new, unseen environment.

This is a critical test because rapid context switching is notoriously difficult for LLMs, which often struggle with "contextual inertia"—the tendency to carry over patterns or assumptions from previous data into a new, unrelated environment.

## Precise Execution Steps

1. **Warming Up (Stream A)**:
   - The AGI observes "Stream A" for an initial baseline context window of 30 seconds.

2. **The Progressive Switch**:
   - The model is subjected to a "Blind Switch" where the context buffer begins to transition from Stream A to Stream B.
   - The benchmark performs an incremental **Progressive Polling Loop** in **5-second steps**:
     - For $t = 5, 10, 15, 20, 25, 30$ seconds:
       - The AGI is provided with the 30s of Stream A followed by $t$ seconds of Stream B.
       - The AGI is asked to predict the chat messages for the NEXT 10 seconds of Stream B.
       - The prediction is evaluated by the Judge LLM.
       - If the model successfully passes all criteria (proves it has pivoted its world model), the loop breaks and $t$ is recorded as the **Adaptation Latency**.

3. **Detection vs. Notification**:
   - The model is **not** explicitly notified that a switch occurred. It must autonomously detect the change from the mixed video and chat context.

## Automated Ground Truth
The ground truth for each probe is the actual synchronized chat logs gathered from "Stream B" immediately following the current $t$-second transition window.

## Evaluation Metrics

To pass each probe, the generated prediction is evaluated using an **LLM-as-a-Judge** (`kbench.assertions.assess_response_with_judge`). 

The evaluation criteria are:
1. **Contextual Recognition**: The model successfully recognized the stream change and did not hallucinate data from Stream A.
2. **Social Adaptation**: The prediction accurately reflects the genre, tone, and social dynamics of Stream B.
3. **Semantic Validity**: The prediction is semantically valid for the new ground truth chat.

### Scoring Logic
The final score for the switch is calculated on a linear scale based on how quickly the model adapted:
- **Immediate Adaptation ($t=5s$):** 100 points.
- **Max Latency/Failure ($t=30s$):** 0 points.
- **Formula:** $Score = 100 \times (1 - \frac{t - 5}{25})$

## Implementation Note
A successful AGI should demonstrably prove it is no longer relying on the semantic priors established during Stream A within the first few seconds of the transition.
