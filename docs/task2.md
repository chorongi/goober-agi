# Task 2: Past Frame Generation (Visual State Description from Chat)

**Description**: The AGI must reconstruct the visual state of a withheld 10-second video segment by providing a detailed text description, using *only* the accumulated live chat from that same interval. This tests the model's ability to translate human social signals (reactions, descriptions, commands) back into the visual "world state" that triggered them.

## Precise Execution Steps

1. **Context Initialization**: The AGI is fed the stream (Video + Chat) up to time $T$.
2. **Visual Blackout**: From $T$ to $T+10$, the video feed is completely withheld. The AGI only receives the stream of chat messages $C_{T \to T+10}$.
3. **Internal Simulation**: The AGI correlates the specific sentiment, keywords, and frequency of the chat messages with its learned internal model of the stream (e.g., recognizing that "F in the chat" usually follows a character death or a failure).
4. **Textual Generation**: The AGI generates a text description predicting the visual state for the 10 withheld frames (one for each second).

## Evaluation Metrics

To pass this task, the generated text description is evaluated using an **LLM-as-a-Judge** (`kbench.assertions.assess_response_with_judge`). The Judge compares the predicted visual description against the withheld chat context to determine if the AI successfully "saw" the video through the eyes of the chat.

The evaluation criteria are:
1. **Semantic Alignment**: The reconstructed descriptions align with the events described in the withheld chat.
2. **Temporal Consistency**: The temporal sequence of events matches the chat's reaction timing.
3. **Contextual Plausibility**: The visual descriptions are plausible for the context of this specific stream.

## Implementation Note
A successful AGI should not only produce a plausible description but one that captures the *specific event* described in the chat (e.g., if the chat says "GG", the text description should likely mention a 'Game Over' screen or a victory pose).
