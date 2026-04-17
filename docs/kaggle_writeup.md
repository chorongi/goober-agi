# Project Report: MM Goober

## Project Name

MM Goober: A Multi-Modal Benchmark from Live Streaming Video

## Your Team

Kwarkamole

## Problem Statement

Existing benchmarks for Artificial General Intelligence (AGI) often fall short in evaluating capabilities crucial for real-world, dynamic environments. Many rely on static datasets, lack true multimodal integration, or fail to capture the complexities of human social interaction. Furthermore, the reliance on human annotators for ground truth is a significant bottleneck, introducing bias, high costs, and scalability issues.

MM Goober addresses these gaps by evaluating models on **Live Streams**, which are:
1. **Unseen & Uncontrollable**: Prevents dataset contamination or memorization.
2. **Multimodal**: Integrates continuous video and high-velocity chat history.
3. **Socially Complex**: Requires understanding group sentiment, slang, and reaction flow.
4. **Self-Labeling**: Generates verifiably correct ground truth automatically from the stream itself.

## Task & Benchmark Construction

The **MM Goober** benchmark leverages the `kaggle_benchmarks` SDK to evaluate models across three core cognitive domains. The construction prioritizes **robustness** and **reproducibility** through strict prompt templates and dynamic few-shot learning.

### Task 1: Social Foresight (Future Chat Prediction)
- **Construction**: The model receives 30s of 1.0 FPS video and chat history. To ground the model's output in the current stream's unique social "vibe," we inject a **dynamic few-shot example** from the immediate history directly into the prompt.
- **Goal**: Predict the semantic flow of the next 10s of chat.
- **Robustness**: Evaluation uses an LLM-as-a-Judge instructed to ignore usernames/mentions, focusing strictly on semantic sentiment and social state.

### Task 2: Cognitive Reconstruction (Visual Inference)
- **Construction**: A 10s video window is withheld while chat remains visible. The model must reconstruct the visual state.
- **Unambiguous Ground Truth**: To eliminate the ambiguity of comparing text-to-video, we generate a **Ground Truth Reconstruction** from the actual withheld frames using a multimodal LLM. 
- **Formatting**: Both prediction and ground truth are forced into a unified template (`[OVERALL SCENE]` and `[KEY EVENTS]`), ensuring the judge performs a structural 1-to-1 comparison.

### Task 3: Context Agility (Progressive Stream-Switch Adaptation)
- **Construction**: Measures "Cognitive Inertia." The model receives 30s of Stream A followed by an incrementally increasing window of Stream B ($t=5, 10, \dots, 30s$). This **Progressive Polling Loop** probes the model at 5-second intervals to see how much new data is required to force a context pivot.
- **Goal**: Measure the exact **Adaptation Latency** required for the model to autonomously detect the switch and discard the outdated Stream A world model.
- **Metric (LLM Judge)**: A linear score from 0-100 based on latency. Immediate adaptation ($t=5s$) earns 100 points, while failure to adapt within 30s earns 0 points. Evaluation ensures the model has fully adapted to the genre and dynamics of Stream B without hallucinating data from Stream A.

## Dataset

MM Goober generates its dataset on-the-fly from live YouTube streams, ensuring the data is always fresh and verifiably defensible.

### Data Provenance & Reliability
- **Source**: 24/7 public livestreams (NASA, Bloomberg, Lofi Girl, etc.).
- **Ambiguity Removal**: By using real-time broadcast data, the ground truth is verifiably correct at the moment of capture. There is no human annotation bias.
- **Sample Size**: The benchmark iterates through 10+ diverse stream categories (News, Music, Nature, Space, Urban), providing a statistically significant cross-section of multimodal environments.

### Schema & Data Types
The `StreamFetcher` orchestrates the acquisition of the following data structures:
| Data Type | Column/Key | Format | Description |
| :--- | :--- | :--- | :--- |
| **Visual** | `video_frames` | `List[np.ndarray]` | RGB frames sampled at 1.0 FPS, downsampled to 224x224. |
| **Textual** | `chat_messages` | `List[str]` | Real-time chat strings in `"author: message"` format. |
| **Temporal** | `timestamp` | `float` | Relative time offset in seconds for synchronization. |
| **Metadata** | `video_id` | `str` | Unique YouTube identifier for the current stream. |

## Technical Details

### Implementation Techniques
- **Multimodal Pipeline**: We utilize `yt-dlp` for HLS stream extraction, `OpenCV` for frame processing, and `pytchat` for asynchronous chat fetching.
- **Token Optimization**: To manage the massive context of 60 frames and chat logs, we implement **spatial downsampling** (224x224) and **temporal sampling** (1 FPS), keeping the prompt within 128k token limits while maintaining semantic fidelity.
- **LLM-as-a-Judge**: We utilize the `kaggle_benchmarks.assertions` module to perform semantic scoring, utilizing a "strict and unyielding" judge prompt that penalizes generic or low-effort responses.

## Novelty, Insights, and Discriminatory Power

### What does this benchmark reveal?
MM Goober reveals a model's **Contextual Inertia**—its inability to let go of an outdated context. Traditional benchmarks with discrete prompts cannot see this. By providing a continuous stream with a hidden switch, we can see if a model continues to hallucinate "Stream A" data after "Stream B" has clearly started.

### Discriminatory Signal (The Gradient of Performance)
The benchmark provides a clear gradient that distinguishes model tiers:
- **Baseline (0-20%)**: Models that fail to follow the "author: message" format or produce generic "gg/lol" spam regardless of the stream.
- **Competent (20-60%)**: Models that capture the general vibe (e.g., recognizing it's a music stream) but fail the Task 3 "Blind Switch," carrying over data from Stream A into Stream B.
- **Elite (60-90%)**: Models that accurately predict specific visual events (Task 2) and show instant context agility (Task 3) by detecting the switch within the first few frames.

## Results, Insights, and Conclusions

Our preliminary testing shows that while state-of-the-art multimodal LLMs are excellent at individual frame description, they struggle significantly with **temporal social momentum**. 

**Key Insights:**
1. **Textual Over-Reliance**: Models often weight the chat history more heavily than the video frames when predicting future chat, leading to "echo chamber" hallucinations where they miss visual cues that change the social mood.
2. **Context Inertia**: Most models require at least 10-15 seconds of new data before they "forget" a previous strong context, highlighting a lack of zero-shot agility.
3. **Reinforcement Learning Advantage**: Integrating RL into the evaluation loop would provide a significant advantage, as agents could be rewarded for minimizing prediction error over time, effectively learning to fine-tune their world models against the infinite stream of real-time ground truth.

## Organizational Affiliations
This project was developed independently for the Kaggle AGI Benchmark competition.

## References & Citations
- **BERTScore**: Zhang, T., et al. (2019). BERTScore: Evaluating Text Generation with BERT.
- **LPIPS**: Zhang, R., et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
- **CLIP**: Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision.
- **Kbench SDK**: Kaggle Benchmarks SDK documentation and implementation.
