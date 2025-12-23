import math
import torch

def greedy_decode(vocab: str, logits: torch.Tensor) -> list[str]:
    """
    Simple decoder for debugging:
    - argmax over classes
    - collapse repeats
    - remove blank tokens
    """
    # logits: (T, B, num_classes)
    preds = logits.argmax(2)  # (T, B)
    preds = preds.cpu().numpy()

    vocab_len = len(vocab)
    blank_index = len(vocab)

    results = []
    for b in range(preds.shape[1]):
        seq = preds[:, b]
        prev = blank_index
        text = ""

        for idx in seq:
            if idx != prev and idx != blank_index:
                if 0 <= idx < vocab_len:
                    text += vocab[idx]
            prev = idx

        results.append(text)

    return results

def greedy_decode_single(vocab: str, logits: torch.Tensor) -> str:
    """Greedy CTC decode for a single example.

    Args:
        logits: Tensor of shape (T, B=1, C=num_classes)
    Returns:
        Decoded string
    """
    preds = logits.argmax(2).squeeze(1).tolist()  # (T,)

    vocab_len = len(vocab)
    blank_index = vocab_len

    text: list[str] = []
    prev = blank_index
    for idx in preds:
        if idx != prev and idx != blank_index:
            if 0 <= idx < len(vocab):
                text.append(vocab[idx])
        prev = idx
    return "".join(text)

def greedy_decode_with_confidence(alphabet: str, logits: torch.Tensor) -> tuple[str, float]:
    """Returns (prediction, confidence)."""
    # Handle (timesteps, batch, classes) -> (batch, timesteps, classes)
    # I might change this in the future so only reshape if the model has
    # more dimensions in 0 than in 1
    if logits.shape[0] > logits.shape[1]:
        logits = logits.permute(1, 0, 2)

    # logits shape: (1, timesteps, num_classes)
    probs = logits.softmax(dim=-1)  # convert to probabilities
    
    max_probs, indices = probs.max(dim=-1)  # (1, timesteps)
    
    # Decode characters
    chars = []
    char_probs = []
    prev_idx = -1
    blank_idx = len(alphabet)
    
    for t in range(indices.shape[1]):
        idx = indices[0, t].item()
        prob = max_probs[0, t].item()
        
        # CTC: skip blanks and repeated chars
        if idx != blank_idx and idx != prev_idx:
            chars.append(alphabet[idx])
            char_probs.append(prob)
        prev_idx = idx
    
    prediction = ''.join(chars)
    
    # Overall confidence: product of per-char probs (or min, or mean)
    if char_probs:
        confidence = min(char_probs)  # conservative: weakest link
        # or: confidence = math.prod(char_probs)  # product
        # or: confidence = sum(char_probs) / len(char_probs)  # mean
    else:
        confidence = 0.0
    
    return prediction, confidence

def beam_decode(
    alphabet: str,
    logits: torch.Tensor,
    beam_width: int = 5
) -> list[tuple[str, float]]:
    """
    Beam search decoding for CTC.
    
    Returns list of (prediction, confidence) tuples, sorted by confidence.
    """
    probs = logits.softmax(dim=-1)  # (1, timesteps, num_classes)
    probs = probs.squeeze(0)        # (timesteps, num_classes)
    blank_idx = len(alphabet)

    # Each beam: (sequence, last_char_idx, cumulative_log_prob)
    beams = [("", -1, 0.0)]
    
    for t in range(probs.shape[0]):
        candidates = []
        
        for seq, prev_idx, cum_log_prob in beams:
            for idx in range(probs.shape[1]):
                prob = probs[t, idx].item()
                
                if prob < 0.001:  # prune unlikely paths early
                    continue
                
                log_prob = math.log(prob + 1e-10)
                new_cum = cum_log_prob + log_prob
                
                if idx == blank_idx:
                    # Blank: sequence unchanged
                    candidates.append((seq, idx, new_cum))
                elif idx == prev_idx:
                    # Repeat: sequence unchanged (CTC collapse)
                    candidates.append((seq, idx, new_cum))
                else:
                    # New character
                    new_seq = seq + alphabet[idx]
                    candidates.append((new_seq, idx, new_cum))
        
        # Merge beams with same sequence, keep best prob for each
        merged: dict[tuple[str, int], float] = {}
        for seq, prev_idx, cum_log_prob in candidates:
            key = (seq, prev_idx)
            if key not in merged or cum_log_prob > merged[key]:
                merged[key] = cum_log_prob
        
        # Keep top beam_width
        sorted_beams = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        beams = [(seq, prev_idx, log_prob) for (seq, prev_idx), log_prob in sorted_beams[:beam_width]]
    
    # Convert log probs back to confidence and dedupe sequences
    results: dict[str, float] = {}
    for seq, _, cum_log_prob in beams:
        confidence = math.exp(cum_log_prob)
        if seq not in results or confidence > results[seq]:
            results[seq] = confidence
    
    # Sort by confidence
    return sorted(results.items(), key=lambda x: x[1], reverse=True)

class DecodeUtils:
    @classmethod
    def greedy_decode(cls, alphabet: str, logits: torch.Tensor) -> list[str]:
        return greedy_decode(alphabet, logits)

    @classmethod
    def greedy_decode_single(cls, alphabet: str, logits: torch.Tensor) -> str:
        return greedy_decode_single(alphabet, logits)
    
    @classmethod
    def greedy_decode_with_confidence(cls, alphabet: str, logits: torch.Tensor) -> tuple[str, float]:
        return greedy_decode_with_confidence(alphabet, logits)
    
    @classmethod
    def beam_decode(cls, alphabet: str, logits: torch.Tensor, beam_width: int = 5) -> list[tuple[str, float]]:
        return beam_decode(alphabet, logits, beam_width)