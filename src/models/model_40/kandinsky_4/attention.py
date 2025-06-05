import torch
import torch.nn.functional as F
import math


def standard_flash_attn_varlen_qkvpacked_func_replacement(
    query_key_value,
    large_cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
):
    device = query_key_value.device
    dtype = query_key_value.dtype

    if query_key_value.dim() == 4:
        if query_key_value.shape[1] == 3:
            q, k, v = query_key_value.unbind(1)
        elif query_key_value.shape[2] == 3:
            q, k, v = query_key_value.unbind(2)
        else:
            raise ValueError(
                f"Unexpected query_key_value shape: {query_key_value.shape}"
            )
    else:
        raise ValueError(f"Unexpected query_key_value shape: {query_key_value.shape}")

    total_tokens, num_heads, head_dim = q.shape
    batch_size = len(large_cu_seqlens) - 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    output = torch.zeros(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    lse_output = torch.zeros(total_tokens, num_heads, device=device, dtype=dtype)

    attn_probs_output = None

    operation_count = 0
    cache_clear_interval = 50

    for batch_idx in range(batch_size):
        start_idx = large_cu_seqlens[batch_idx].item()
        end_idx = large_cu_seqlens[batch_idx + 1].item()
        seq_len = end_idx - start_idx

        if seq_len == 0:
            continue

        query_chunk_size = min(32, max(8, seq_len // 100))

        head_group_size = min(4, num_heads)

        for head_start in range(0, num_heads, head_group_size):
            head_end = min(head_start + head_group_size, num_heads)
            current_head_count = head_end - head_start

            q_heads = q[start_idx:end_idx, head_start:head_end, :]
            k_heads = k[start_idx:end_idx, head_start:head_end, :]
            v_heads = v[start_idx:end_idx, head_start:head_end, :]

            for q_start in range(0, seq_len, query_chunk_size):
                q_end = min(q_start + query_chunk_size, seq_len)
                chunk_size = q_end - q_start

                q_chunk = q_heads[q_start:q_end]

                scores = torch.einsum("qhd,khd->qhk", q_chunk, k_heads) * softmax_scale

                if causal:
                    mask = torch.triu(
                        torch.ones(
                            chunk_size, seq_len, device=device, dtype=torch.bool
                        ),
                        diagonal=q_start + 1,
                    )
                    scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

                lse = torch.logsumexp(scores, dim=-1)

                attn_weights = torch.exp(scores - lse.unsqueeze(-1))

                if dropout_p > 0.0 and torch.is_grad_enabled():
                    attn_weights = F.dropout(attn_weights, p=dropout_p)

                chunk_output = torch.einsum("qhk,khd->qhd", attn_weights, v_heads)

                global_q_start = start_idx + q_start
                global_q_end = start_idx + q_end
                output[global_q_start:global_q_end, head_start:head_end, :] = (
                    chunk_output
                )
                lse_output[global_q_start:global_q_end, head_start:head_end] = lse

                del scores, lse, attn_weights, chunk_output

                operation_count += 1
                if operation_count % cache_clear_interval == 0:
                    torch.cuda.empty_cache()

            del q_heads, k_heads, v_heads

    output = output.contiguous()
    lse_output = lse_output.contiguous()

    torch.cuda.empty_cache()

    return output, lse_output, attn_probs_output
