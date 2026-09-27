from __future__ import annotations

from collections import Counter

import torch

try:
    import spaces
except Exception:
    class _Spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    spaces = _Spaces()

from .metrics import glyph_stream


def _duration_for_probe(text: str, window: int, max_symbols: int) -> int:
    n = min(len(glyph_stream(text)), int(max_symbols))
    return int(max(10, min(45, 10 + n / 5000)))


@spaces.GPU(duration=_duration_for_probe)
def gpu_cooccurrence_probe(text: str, window: int = 5, max_symbols: int = 50000):
    stream = glyph_stream(text)[: int(max_symbols)]
    if len(stream) < 4:
        raise ValueError("Need at least 4 glyphs for the GPU probe.")
    vocab = sorted(set(stream))
    index = {ch: i for i, ch in enumerate(vocab)}
    ids = torch.tensor([index[ch] for ch in stream], dtype=torch.long, device="cuda")
    v = len(vocab)
    co = torch.zeros((v, v), dtype=torch.float32, device="cuda")
    w = max(1, min(int(window), 64))
    for offset in range(1, w + 1):
        if len(ids) <= offset:
            break
        a = ids[:-offset]
        b = ids[offset:]
        flat = a * v + b
        counts = torch.bincount(flat, minlength=v * v).reshape(v, v)
        co += counts
        co += counts.T
    total = co.sum().clamp_min(1)
    pxy = co / total
    px = pxy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    py = pxy.sum(dim=0, keepdim=True).clamp_min(1e-12)
    ppmi = torch.clamp(torch.log(pxy.clamp_min(1e-12) / (px @ py)), min=0)
    _u, s, _vh = torch.linalg.svd(ppmi)
    top = s[: min(10, len(s))].detach().cpu().tolist()
    freq = Counter(stream)
    return {"glyphs_used": len(stream), "vocab_size": v, "window": w, "top_singular_values": [float(x) for x in top], "device": str(ids.device), "note": "Exploratory representation probe only; not evidence of language or decipherment.", "glyph_table": [{"glyph": ch, "count": freq[ch], "index": index[ch]} for ch in sorted(vocab, key=lambda c: (-freq[c], c))][:100]}
