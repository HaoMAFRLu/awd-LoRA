"""lm-evaluation-harness 0.4.9.1 loglikelihood adapter for this architecture."""
from __future__ import annotations

import torch
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from .trainer import compute_context


class MoEHarnessLM(TemplateLM):
    def __init__(self, model, tokenizer, config, device):
        super().__init__()
        self.model, self.tokenizer, self.config = model.eval(), tokenizer, config
        self.device = torch.device(device)
        self.max_length = config["data"]["seq_length"]
        self.batch_size = 1  # explicit unpadded reference scorer

    @property
    def eot_token_id(self):
        return self.config["data"]["eod_id"]

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)

    @torch.no_grad()
    def score_tokens(self, context, continuation):
        if not continuation:
            return 0.0, True
        if len(continuation) > self.max_length:
            raise ValueError(
                "A continuation exceeds the context window; use rolling likelihood for long documents"
            )
        context = context or [self.eot_token_id]
        tokens = (context + continuation)[-(self.max_length + 1) :]
        inputs = torch.tensor(tokens[:-1], device=self.device, dtype=torch.long)[None]
        targets = torch.tensor(continuation, device=self.device, dtype=torch.long)
        with compute_context(self.config, self.device):
            output = self.model(inputs)
        logits = output.logits[0, -len(continuation) :].float()
        log_probs = logits.log_softmax(-1).gather(-1, targets[:, None]).sum().item()
        return log_probs, bool(torch.equal(logits.argmax(-1), targets))

    def _loglikelihood_tokens(self, requests, **kwargs):
        results = []
        for key, context, continuation in requests:
            result = self.score_tokens(context, continuation)
            results.append(result)
            if key is not None:
                self.cache_hook.add_partial("loglikelihood", key, result)
        return results

    def loglikelihood_rolling(self, requests, **kwargs):
        results = []
        for request in requests:
            tokens = self.tok_encode(request.args[0])
            windows = get_rolling_token_windows(
                tokens, self.eot_token_id, self.max_length, context_len=1
            )
            results.append(
                sum(self.score_tokens(*make_disjoint_window(window))[0] for window in windows)
            )
        return results

    def generate_until(self, requests, **kwargs):
        raise NotImplementedError(
            "The six configured zero-shot tasks use loglikelihood; generation is outside this adapter's scope"
        )
