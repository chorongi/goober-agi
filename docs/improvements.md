# Future Benchmark Improvements

This document outlines planned upgrades to the CMFSP (Cross-Modal Future State Prediction) benchmark to fully realize its original multimodal design.

## Task 2: True Past Frame Generation (Image vs. Image)

Currently, Task 2 evaluates text descriptions of withheld video frames using an LLM-as-a-judge. The original intent and future goal of this task is to require the AGI to generate actual **RGB image arrays** (a sequence of images) representing the withheld frames, rather than text.

Once image generation models (like diffusion models) are integrated into the benchmark pipeline, the evaluation methodology must be updated to bypass the `kbench` text-based judge and instead use mathematical tensor comparisons. The scaffolding for this is already present in `metrics.py`.

### Planned Evaluation Metrics (Mathematical)

1. **LPIPS (Primary - Perceptual similarity)**:
   - **Why**: Traditional SSIM/PSNR are too sensitive to pixel shifts. LPIPS (Learned Perceptual Image Patch Similarity) uses deep features to measure if the *content* and *layout* of the generated frame match the original ground truth frame.
   - **Target**: Lower is better (0.0 is a perfect match).

2. **CLIP Visual-Text Alignment (Secondary - Semantic consistency)**:
   - **Why**: Since the reconstruction is based on chat, we measure the cosine similarity between the CLIP embeddings of the generated frames and the concatenated chat text to ensure semantic alignment.
   - **Metric**: $cos(\text{CLIP}_{img}(P_i), \text{CLIP}_{txt}(C_{T \to T+10}))$.

3. **Temporal Consistency (Sequence Check)**:
   - Measure the average LPIPS distance between adjacent generated frames ($P_i, P_{i+1}$) and compare it to the ground truth's temporal delta to ensure the model isn't just generating 10 identical static images.

### Required Code Changes
To implement this upgrade, the following changes will be needed:
*   Update the AGI prompt logic in `task2.py` (and the notebook generator) to explicitly request and handle `List[np.ndarray]` image outputs instead of a text string.
*   Remove the `kbench.assertions.assess_response_with_judge` block from Task 2.
*   Import and invoke `Evaluator.evaluate_task2()` from `metrics.py` to calculate the final score by comparing the generated frames to the `ground_truth_frames`.
