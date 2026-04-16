import numpy as np
from typing import List

try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    import torch
    import lpips
    import clip
    from PIL import Image
    METRICS_AVAILABLE = True
except ImportError:
    print(
        "Warning: ML Evaluation libraries (rouge_score, bert_score, lpips, clip) not fully installed. Falling back to mock metrics for structural testing."
    )
    METRICS_AVAILABLE = False
    # Mocking for type checking when not available
    rouge_scorer = None
    bert_score = None
    torch = None
    lpips = None
    clip = None
    Image = None

class Evaluator:
    def __init__(self):
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # type: ignore
            self.device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore
            self.lpips_loss_fn_vgg = lpips.LPIPS(net="vgg").to(self.device)  # type: ignore
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)  # type: ignore

    def evaluate_task1(self, pred_chat: List[str], true_chat: List[str]) -> float:
        """Returns a normalized score (0-100) using BERTScore and ROUGE-L."""
        if not pred_chat or not true_chat:
            return 0.0

        import re

        def clean_chat(chat_list: List[str]) -> str:
            cleaned = []
            for msg in chat_list:
                # Remove "Username: " prefix if present
                if ":" in msg:
                    msg = msg.split(":", 1)[1].strip()
                # Remove "@username" mentions
                msg = re.sub(r"@\w+", "", msg).strip()
                if msg:
                    cleaned.append(msg)
            return " ".join(cleaned)

        pred_text = clean_chat(pred_chat)
        true_text = clean_chat(true_chat)

        if not pred_text or not true_text:
            return 0.0

        if not METRICS_AVAILABLE:
            return float(np.random.uniform(50, 90))  # Mock score

        # 1. ROUGE-L
        rouge_scores = self.rouge_scorer.score(true_text, pred_text)  # type: ignore
        rouge_f1 = rouge_scores["rougeL"].fmeasure

        # 2. BERTScore
        P, R, F1 = bert_score([pred_text], [true_text], lang="en", verbose=False)  # type: ignore
        bert_f1 = float(F1.item())  # type: ignore

        # Weighted combination as defined in task1.md
        final_score = (0.7 * bert_f1) + (0.3 * rouge_f1)
        return float(final_score * 100)

    def evaluate_task2(
        self,
        pred_frames: List[np.ndarray],
        true_frames: List[np.ndarray],
        chat_context: List[str],
    ) -> float:
        """Returns a normalized score (0-100) using LPIPS and CLIP alignment."""
        if not pred_frames or not true_frames:
            return 0.0

        if not METRICS_AVAILABLE:
            return float(np.random.uniform(40, 85))  # Mock score

        # Ensure lists are same length
        min_len = min(len(pred_frames), len(true_frames))
        pred_frames = pred_frames[:min_len]
        true_frames = true_frames[:min_len]

        lpips_scores = []
        clip_scores = []
        chat_text = " ".join(chat_context)
        text_input = clip.tokenize([chat_text[:77]]).to(self.device)  # type: ignore

        with torch.no_grad():  # type: ignore
            for p_img, t_img in zip(pred_frames, true_frames):
                # Convert to PIL then to torch tensors
                p_pil = Image.fromarray(p_img)  # type: ignore

                # 1. LPIPS Calculation
                # LPIPS expects tensors in [-1, 1]
                p_tensor = (
                    torch.tensor(p_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5  # type: ignore
                    - 1
                )
                t_tensor = (
                    torch.tensor(t_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5  # type: ignore
                    - 1
                )

                loss = self.lpips_loss_fn_vgg(
                    p_tensor.to(self.device), t_tensor.to(self.device)
                )
                lpips_scores.append(float(loss.item()))  # type: ignore

                # 2. CLIP Alignment Calculation
                image_input = self.clip_preprocess(p_pil).unsqueeze(0).to(self.device)  # type: ignore
                image_features = self.clip_model.encode_image(image_input)  # type: ignore
                text_features = self.clip_model.encode_text(text_input)  # type: ignore

                # Cosine similarity
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).item()
                clip_scores.append(float(similarity))

        # Normalize LPIPS (0 is best, usually maxes around 1.0) -> Convert to 0-100 where 100 is best
        avg_lpips = np.mean(lpips_scores)
        lpips_score_normalized = max(0.0, (1.0 - float(avg_lpips))) * 100

        # Normalize CLIP (-1 to 1) -> Convert to 0-100
        avg_clip = np.mean(clip_scores)
        clip_score_normalized = ((float(avg_clip) + 1) / 2) * 100

        # Final Task 2 Score (equal weight perceptual and semantic)
        final_score = (0.5 * lpips_score_normalized) + (0.5 * clip_score_normalized)
        return float(final_score)
