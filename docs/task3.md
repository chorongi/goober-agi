# Task 3: Stream-Switch Adaptation (Context Agility)

**Description**: This task measures the "cognitive agility" of the AGI. It quantifies the model's ability to perform a **Context Switch**: discarding an outdated world model and instantly reaching a functional level of predictive accuracy in a completely new, unseen environment.

This is a critical test because rapid context switching is notoriously difficult for LLMs, which often struggle with "contextual inertia"—the tendency to carry over patterns or assumptions from previous data into a new, unrelated environment.

## Precise Execution Steps

1. **Warming Up (Stream A)**:
   - The AGI observes "Stream A" for an initial context window (e.g., 30 seconds).

2. **The Hard Switch**:
   - The input feed is instantly swapped from "Stream A" to "Stream B" (a different genre, game, or creator).
   - The AGI is given a brief, 5-second "glance" of the new stream (video and chat) to orient itself.
   - The AGI is explicitly notified: *"CONTEXT SWITCH: You have just been switched to a NEW stream."*

3. **Zero-Shot Prediction**:
   - Based *only* on the 5-second glance, the AGI must immediately predict the chat messages for the NEXT 10 seconds of Stream B.

## Automated Ground Truth
The ground truth is the actual synchronized chat logs gathered from "Stream B" immediately following the 5-second glance window.

## Evaluation Metrics

To pass this task, the generated zero-shot prediction is evaluated using an **LLM-as-a-Judge** (`kbench.assertions.assess_response_with_judge`). The Judge compares the predicted chat against the actual 10-second ground truth chat of the new stream.

The evaluation criteria are:
1. **Contextual Recognition**: The model successfully recognized the context switch (did not hallucinate Stream A data).
2. **Social Adaptation**: The zero-shot prediction shows understanding of the new stream's genre and social dynamics.
3. **Semantic Validity**: The prediction is semantically valid for the new ground truth.

For each stream switch pair, the score is determined by the percentage of criteria passed via the Judge LLM.

## Implementation Note
A successful AGI should not only produce valid predictions for Stream B but must also demonstrably prove it is no longer relying on the semantic priors established during its time observing Stream A.
