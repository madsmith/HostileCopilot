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