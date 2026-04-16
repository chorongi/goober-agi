# Task 2: Past Frame Generation (Visual Reconstruction from Chat)

**Description**: The AGI must reconstruct a withheld 10-second video segment at 1 FPS, using *only* the accumulated live chat from that same interval. This tests the model's ability to translate human social signals (reactions, descriptions, commands) back into the visual "world state" that triggered them.

## Precise Execution Steps

1. **Context Initialization**: The AGI is fed the stream (Video + Audio + Chat) up to time $T$.
2. **Visual Blackout**: From $T$ to $T+10$, the video and audio feeds are completely withheld. The AGI only receives the stream of chat messages $C_{T \to T+10}$ in real-time or as a batch.
3. **Internal Simulation**: The AGI correlates the specific sentiment, keywords, and frequency of the chat messages with its learned internal model of the stream (e.g., recognizing that "F in the chat" usually follows a character death or a failure).
4. **Synchronized Generation**: The AGI generates 10 discrete RGB images ($P_1, P_2, \dots, P_{10}$), where $P_i$ is the predicted visual state at exactly $T + i$ seconds.

## Automated Ground Truth
The ground truth consists of the 10 actual video frames ($G_1, G_2, \dots, G_{10}$) extracted from the live stream at 1 FPS during the withheld interval $[T+1, T+10]$.

## Evaluation Metrics

To pass this task, the generated sequence is evaluated against the ground truth using a three-tier metric system:

1. **LPIPS (Primary - Perceptual similarity)**:
   - **Why**: Traditional SSIM/PSNR are too sensitive to pixel shifts. LPIPS (Learned Perceptual Image Patch Similarity) uses deep features to measure if the *content* and *layout* of the generated frame match the original.
   - **Target**: Lower is better (0.0 is a perfect match).

2. **CLIP Visual-Text Alignment (Secondary - Semantic consistency)**:
   - **Why**: Since the reconstruction is based on chat, we measure the cosine similarity between the CLIP embeddings of the generated frames and the concatenated chat text. 
   - **Metric**: $cos(\text{CLIP}_{img}(P_i), \text{CLIP}_{txt}(C_{T \to T+10}))$.

3. **Temporal Consistency (Sequence Check)**:
   - We measure the average LPIPS distance between adjacent generated frames ($P_i, P_{i+1}$) and compare it to the ground truth's temporal delta to ensure the model isn't just generating 10 identical static images.

## Implementation Note
A successful AGI should not only produce a visually plausible image but one that captures the *specific event* described in the chat (e.g., if the chat says "GG", the generated frame should likely show a 'Game Over' screen or a victory pose).
