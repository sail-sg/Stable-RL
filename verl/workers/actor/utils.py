import torch

@torch.no_grad()
@torch.compile(fullgraph=True)
def compute_topk_kl(logits, rollout_topk_token_ids, rollout_topk_logprobs):
    """
    Computes KL divergence between the predicted logits (Q) and a "teacher" 
    distribution (P), given as rollout_topk_token_ids and rollout_topk_logprobs,
    in a 'top-k + aggregated rest' scheme:
      - For each position: retain explicit KL contributions for the top-k tokens.
      - Aggregate all other ("rest") token probabilities/logits into a single token.

    Args:
        logits: Tensor of shape (..., vocab_size)
        rollout_topk_token_ids: IntTensor of shape (..., k)
        rollout_topk_logprobs: FloatTensor of shape (..., k)
    Returns:
        kl: Tensor of shape (...), KL divergence for each sample in the batch/sequence
    """
    logits = logits.to(torch.float32)
    rollout_topk_token_ids = rollout_topk_token_ids.to(torch.int32)
    rollout_topk_logprobs = rollout_topk_logprobs.to(torch.float32)

    # Q for topk tokens
    log_q = torch.log_softmax(logits, dim=-1)             # (..., vocab_size)
    topk_q_log_probs = torch.gather(log_q, -1, rollout_topk_token_ids)  # (..., k)
    topk_q_probs = torch.exp(topk_q_log_probs)
    topk_q_mass = topk_q_probs.sum(dim=-1, keepdim=True)               # (..., 1)
    rest_q_mass = 1.0 - topk_q_mass

    topk_p_probs = torch.exp(rollout_topk_logprobs)                    # (..., k)
    topk_p_mass = topk_p_probs.sum(dim=-1, keepdim=True)               # (..., 1)
    rest_p_mass = 1.0 - topk_p_mass                                    # (..., 1)

    # Clamp for numerical stability
    rest_p_mass = rest_p_mass.clamp_(min=1e-12)
    topk_p_mass = topk_p_mass.clamp_(min=1e-12)
    topk_q_mass = topk_q_mass.clamp_(min=1e-12)
    rest_q_mass = rest_q_mass.clamp_(min=1e-12)

    # KL(P||Q) = sum_i p_i * (log(p_i) - log(q_i))
    topk_kl = topk_p_probs * (rollout_topk_logprobs - topk_q_log_probs) # (..., k)
    rest_p_log = rest_p_mass.log()
    rest_q_log = rest_q_mass.log()
    rest_kl = rest_p_mass * (rest_p_log - rest_q_log)
    final_kl = topk_kl.sum(dim=-1) + rest_kl.squeeze(-1)

    topk_tv = torch.abs(topk_p_probs - topk_q_probs).sum(dim=-1)
    rest_tv = torch.abs(rest_p_mass - rest_q_mass).sum(dim=-1)
    final_tv = (topk_tv + rest_tv) / 2.0
    return final_kl, final_tv
