# Task 1: Future Chat Prediction

**Description**: Predict the exact sequence of chat messages that will appear in the live chat during the interval $[T+1, T+10]$ seconds.

**Automated Ground Truth**: The actual chat messages that appear in the live stream between $T+1$ and $T+10$ seconds serve as the ground truth ($G$). Predictions ($P$) and Ground Truth ($G$) are concatenated into single strings for comparison.

## Evaluation Metrics

To effectively measure the quality of chat predictions, we use a hybrid approach that balances semantic meaning with structural overlap:

1. **BERTScore (Primary Metric)**:
   - **Why**: Traditional metrics (BLEU/ROUGE) fail on chat because "lol he died" and "lmfao he got killed" have zero word overlap but identical meaning. BERTScore uses contextual embeddings to measure semantic similarity.
   - **Calculation**: Similarity is calculated using a pre-trained `bert-base-uncased` or `roberta-large` model.

2. **ROUGE-L (Secondary Metric)**:
   - **Why**: Measures the Longest Common Subsequence (LCS). This captures the structural "flow" of the chat and ensures the AGI is capturing the correct sequence of reactions even if the exact vocabulary varies slightly.

3. **Perplexity (Optional/Internal)**:
   - Used during training to measure the model's confidence in the predicted chat distribution.

## Implementation Note
For the final benchmark score, we recommend a weighted average: 
**Score = 0.7 * (BERTScore F1) + 0.3 * (ROUGE-L F1)**
