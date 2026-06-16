"""Unified molecule sampling with quality-guided planning.

Supports 4 quality modes and optional RND (importance weight) computation.

Quality modes:
    "none"            - No planner, no remasking (policy-only)
    "both"            - Both unmasking + insertion planners active
    "unmasking_only"  - Only unmasking/remasking planner (insertion planner disabled)
    "insertion_only"  - Only insertion planner (unmasking planner disabled)

RND toggle:
    compute_rnd=True  - Run pretrained model in parallel, compute step-wise log importance weights
    compute_rnd=False - Run policy model only (use with ELBO-based RND or eval)
"""

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sampling import SamplingResult, SamplingTraceDatapoint, _sample_tokens
from remasking_scheduleaware import apply_schedule_aware_remasking, apply_schedule_aware_insertion
from mol_utils.utils_chem import batch_safe_to_smiles, batch_validate_and_extract
from tdc import Evaluator, Oracle

QUALITY_MODES = {"none", "both", "unmasking_only", "insertion_only"}


@torch.no_grad()
def _diffusion_loop(
    model, steps, mask, pad, batch_size, max_length,
    quality_mode="both",
    compute_rnd=False,
    pretrained=None,
    remasking_mode="schedule_aware",
    num_remasking=1,
    quality_threshold=1,
    temperature=1.0,
    return_trace=False,
    unmask_quality_threshold=None,
):
    """Core discrete diffusion sampling loop for molecule generation.

    Args:
        model: Finetuned policy model.
        steps: Number of diffusion steps.
        mask: Mask token ID.
        pad: Pad token ID.
        batch_size: Number of sequences to generate.
        max_length: Maximum sequence length.
        quality_mode: One of "none", "both", "unmasking_only", "insertion_only".
        compute_rnd: Whether to compute step-wise log importance weights.
        pretrained: Frozen pretrained model (required if compute_rnd=True).
        remasking_mode: Remasking strategy ("schedule_aware", "remdm", "remdm_conf").
        num_remasking: Number of tokens to remask per step.
        quality_threshold: Threshold for insertion quality filtering. None if schedule-driven.
        temperature: Sampling temperature (1.0 = no scaling).
        return_trace: Whether to record sampling trace.

    Returns:
        (xt, log_rnd, sampling_trace)
        log_rnd is None when compute_rnd=False.
    """
    assert quality_mode in QUALITY_MODES, f"quality_mode must be one of {QUALITY_MODES}"
    if compute_rnd:
        assert pretrained is not None, "pretrained model required when compute_rnd=True"

    # Derive flags from quality_mode
    use_remasking = quality_mode != "none"
    disable_unmasking_planner = quality_mode in ("none", "insertion_only")
    disable_insertion_planner = quality_mode in ("none", "unmasking_only")

    device = next(model.parameters()).device

    # Initialize all-pad sequence
    xt = torch.full((batch_size, max_length), pad, dtype=torch.int64, device=device)

    dt = 1.0 / steps
    t = torch.zeros(batch_size, device=device)

    # Precompute index tensors
    batch_idx_L = (
        torch.arange(batch_size, device=device)
        .view(batch_size, 1)
        .expand(batch_size, max_length)
    )
    pos_idx_L = (
        torch.arange(max_length, device=device)
        .view(1, max_length)
        .expand(batch_size, max_length)
    )
    sampling_trace = [[] for _ in range(batch_size)] if return_trace else None

    neg_inf = torch.tensor(-np.inf, device=device)

    if use_remasking and remasking_mode == "remdm_conf":
        remasking_score = torch.zeros((batch_size, max_length), device=device)

    log_rnd = None

    for i in range(steps):
        # --- Policy model forward ---
        pred_rate = model(xt, t)
        pred_rate = model.interpolant.to_actual_rate(xt, pred_rate, t)
        unmask_rate = pred_rate.unmask_rate  # (B, L, V)
        len_rate = pred_rate.length_rate  # (B, L+1)

        # --- Pretrained model forward (for RND) ---
        if compute_rnd:
            pretrained_pred = pretrained(xt, t)
            pretrained_rate = pretrained.interpolant.to_actual_rate(xt, pretrained_pred, t)
            pretrained_unmask_rate = pretrained_rate.unmask_rate.clone()  # (B, L, V)
            pretrained_len_rate = pretrained_rate.length_rate  # (B, L+1)

        # --- Unmask step (Euler) ---
        mask_pos = (xt == mask).nonzero(as_tuple=True)
        unmask_rate[xt != mask] = 0
        unmask_rate[mask_pos + (mask,)] = 0
        unmask_rate[mask_pos + (mask,)] = -unmask_rate[mask_pos + (slice(None),)].sum(dim=1)
        trans_prob = (unmask_rate * dt).clamp(0.0, 1.0)

        if compute_rnd:
            pretrained_unmask_rate[xt != mask] = 0
            pretrained_unmask_rate[mask_pos + (mask,)] = 0
            pretrained_unmask_rate[mask_pos + (mask,)] = -pretrained_unmask_rate[mask_pos + (slice(None),)].sum(dim=1)
            pretrained_trans_prob = (pretrained_unmask_rate * dt).clamp(0.0, 1.0)

        # Add "stay" probability
        _xt = xt.clone()
        _xt[xt == pad] = mask
        trans_prob.scatter_add_(
            2,
            _xt.unsqueeze(-1),
            torch.ones_like(_xt.unsqueeze(-1), dtype=trans_prob.dtype),
        )
        if compute_rnd:
            pretrained_trans_prob.scatter_add_(
                2,
                _xt.unsqueeze(-1),
                torch.ones_like(_xt.unsqueeze(-1), dtype=pretrained_trans_prob.dtype),
            )

        # Temperature scaling
        if temperature != 1.0:
            logits = torch.log(trans_prob + 1e-10) / temperature
            trans_prob = torch.softmax(logits, dim=-1)

        # Final step: remove mask token from sampling
        if i == steps - 1:
            print("Final step, removing mask token from sampling")
            trans_prob[mask_pos + (mask,)] = 0.0

            prob_sum = trans_prob[mask_pos].sum(dim=-1, keepdim=True)
            mask_has_zero_prob = (prob_sum.squeeze(-1) == 0.0)
            if mask_has_zero_prob.any():
                num_zero_prob = mask_has_zero_prob.sum().item()
                uniform_prob = torch.zeros((num_zero_prob, trans_prob.shape[-1]), device=device, dtype=trans_prob.dtype)
                uniform_prob[:, :mask] = 1.0 / mask
                trans_prob[mask_pos[0][mask_has_zero_prob], mask_pos[1][mask_has_zero_prob]] = uniform_prob
            else:
                trans_prob[mask_pos] = trans_prob[mask_pos] / prob_sum

        new_xt = _sample_tokens(trans_prob)
        new_xt[xt == pad] = pad
        new_xt = torch.where((xt != mask) & (xt != pad), xt, new_xt)

        # Update remasking_score buffer for remdm_conf mode
        if use_remasking and remasking_mode == "remdm_conf" and i < steps - 1:
            token_probs = F.softmax(unmask_rate, dim=-1)  # (B, L, V)
            chosen_probs = torch.gather(token_probs, dim=-1, index=new_xt.unsqueeze(-1)).squeeze(-1)  # (B, L)
            changed_mask_to_token = (xt == mask) & (new_xt != mask) & (new_xt != pad)
            remasking_score = torch.where(changed_mask_to_token, chosen_probs, remasking_score)

        # --- Remasking step ---
        if use_remasking and i < steps - 1:
            if disable_unmasking_planner or not (hasattr(model, 'planner') and model.planner is not None):
                remasking_conf = torch.zeros((batch_size, max_length), device=device)
            else:
                planner_out = model.planner(new_xt, t)
                remasking_conf = planner_out["remasking_conf"].squeeze(-1)  # (B, L)

            clean_index = (new_xt != mask) & (new_xt != pad)  # (B, L)

            if remasking_mode == "schedule_aware":
                new_xt = apply_schedule_aware_remasking(
                    model, new_xt, t, dt, remasking_conf, clean_index,
                    mask, neg_inf, batch_size,
                    unmask_quality_threshold=unmask_quality_threshold,
                )
                remasking_score_temp = None
            else:
                raise ValueError(f"Unknown remasking_mode: {remasking_mode}")

            if remasking_score_temp is not None:
                remasking_score_temp = torch.where(clean_index, remasking_score_temp, neg_inf)
                for j in range(batch_size):
                    k = min(num_remasking, int(clean_index[j].sum().item()))
                    if k > 0:
                        _, select_indices = torch.topk(remasking_score_temp[j], k=k)
                        new_xt[j, select_indices] = mask

            if return_trace:
                for batch_idx in range(batch_size):
                    for pos in range(max_length):
                        if clean_index[batch_idx, pos] and new_xt[batch_idx, pos] == mask:
                            sampling_trace[batch_idx].append(
                                SamplingTraceDatapoint(
                                    t=t[batch_idx].item(),
                                    event_type="change",
                                    position=pos,
                                    token=mask,
                                )
                            )

        # --- Compute log probabilities for RND ---
        if compute_rnd:
            lp = torch.gather(torch.log(trans_prob + 1e-10), 2, new_xt.unsqueeze(-1)).squeeze(-1)
            lp_pre = torch.gather(torch.log(pretrained_trans_prob + 1e-10), 2, new_xt.unsqueeze(-1)).squeeze(-1)

            changed_mask = (xt == mask) & (new_xt != mask) & (new_xt != pad)

            log_policy_step = (lp * changed_mask).sum(dim=1)
            log_pretrained_step = (lp_pre * changed_mask).sum(dim=1)

            log_rnd = log_pretrained_step - log_policy_step  # (B,)

        # --- Insertion step ---
        if i != steps - 1:
            ext = torch.poisson(len_rate * dt).long()  # (B, L+1)

            xt_len = xt.ne(pad).sum(dim=1)  # (B,)
            gaps = torch.arange(max_length + 1, device=device).view(1, -1)
            ext = ext * (gaps <= xt_len.view(batch_size, 1)).long()
            total_ext = ext.sum(dim=1)
            valid = xt_len + total_ext <= max_length
            ext = ext * valid.view(batch_size, 1).long()

            ext_ex = ext.int().cumsum(dim=1)  # (B, L+1)
            new_len = xt_len + total_ext  # (B,)

            xt_tmp = torch.full_like(xt, pad)
            mask_fill = pos_idx_L < new_len.view(batch_size, 1)
            xt_tmp[mask_fill] = mask

            new_pos_orig = pos_idx_L + ext_ex[:, :max_length]  # (B, L)
            orig_mask = pos_idx_L < xt_len.view(batch_size, 1)
            flat_b = batch_idx_L[orig_mask]
            flat_p = new_pos_orig[orig_mask]
            xt_tmp[flat_b, flat_p] = new_xt[orig_mask]

            # Schedule-aware insertion quality filtering
            if use_remasking and not disable_insertion_planner:
                if compute_rnd:
                    xt_tmp_before = xt_tmp.clone()

                xt_tmp = apply_schedule_aware_insertion(
                    model, xt_tmp, new_xt, t, dt, ext, mask, pad, max_length,
                    orig_mask, new_pos_orig, quality_threshold
                )

                if compute_rnd:
                    # Compute corrected ext based on what actually stayed
                    ext_corrected = torch.zeros_like(ext)
                    for b in range(batch_size):
                        after_len = xt_tmp[b].ne(pad).sum().item()
                        orig_len = xt_len[b].item()
                        surviving_insertions = after_len - orig_len
                        if total_ext[b] > 0:
                            ratio = surviving_insertions / total_ext[b].item()
                            ext_corrected[b] = (ext[b].float() * ratio).long()
                else:
                    ext_corrected = ext
            else:
                ext_corrected = ext

            # Compute insertion log_rnd
            if compute_rnd:
                insertion_rate = (len_rate * dt).clamp(min=1e-10)  # (B, L+1)
                pretrained_insertion_rate = (pretrained_len_rate * dt).clamp(min=1e-10)  # (B, L+1)

                log_policy_insert = (ext_corrected * torch.log(insertion_rate) - insertion_rate).sum(dim=1)
                log_pretrained_insert = (ext_corrected * torch.log(pretrained_insertion_rate) - pretrained_insertion_rate).sum(dim=1)

                log_insert_diff = log_pretrained_insert - log_policy_insert
                log_rnd += log_insert_diff
        else:
            xt_tmp = new_xt

        if return_trace:
            for batch_idx in range(batch_size):
                for j in range(max_length):
                    if xt[batch_idx, j] != pad and xt[batch_idx, j] != new_xt[batch_idx, j]:
                        sampling_trace[batch_idx].append(
                            SamplingTraceDatapoint(
                                t=t[batch_idx].item(),
                                event_type="change",
                                position=j,
                                token=new_xt[batch_idx, j].item(),
                            )
                        )

                if i != steps - 1:
                    for j in range(max_length):
                        id = max_length - j - 1
                        if ext[batch_idx, id]:
                            sampling_trace[batch_idx].append(
                                SamplingTraceDatapoint(
                                    t=t[batch_idx].item(),
                                    event_type="insertion",
                                    position=id,
                                    token=mask,
                                )
                            )

        xt = xt_tmp
        t = t + dt

    return xt, log_rnd, sampling_trace


def _decode_and_validate(model, tokenizer, samples):
    """Decode token IDs to SMILES and validate.

    Returns:
        (validSequences, valid_indices): list of valid SMILES, list of batch indices.
    """
    decoded_samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

    use_bracket_safe = model.config.training.get('use_bracket_safe', False)
    smiles_samples = batch_safe_to_smiles(decoded_samples, use_bracket_safe=use_bracket_safe, fix=True)

    # Extract valid sequences (take largest fragment)
    validSequences = []
    valid_indices = []
    for idx, s in enumerate(smiles_samples):
        if s:
            largest_frag = sorted(s.split('.'), key=len)[-1]
            validSequences.append(largest_frag)
            valid_indices.append(idx)

    return validSequences, valid_indices


@torch.no_grad()
def sample_mol_buffer(
    model, pretrained, reward_model, tokenizer,
    steps, mask, pad, batch_size, max_length,
    quality_mode="both",
    alpha=0.1,
    remasking_mode="schedule_aware",
    num_remasking=1,
    quality_threshold=1,
    temperature=1.0,
    use_quality_filter=True,
):
    """Generate molecules for training buffer. Always computes step-wise RND.

    Args:
        model: Finetuned policy model.
        pretrained: Frozen pretrained model.
        reward_model: Molecule scoring function.
        tokenizer: SAFE tokenizer for decoding.
        steps: Number of diffusion steps.
        mask: Mask token ID.
        pad: Pad token ID.
        batch_size: Number of sequences to generate.
        max_length: Maximum sequence length.
        quality_mode: "none", "both", "unmasking_only", or "insertion_only".
        alpha: RND scaling factor.
        remasking_mode: Remasking strategy.
        num_remasking: Number of tokens to remask per step.
        quality_threshold: Threshold for insertion quality filtering. None if schedule-driven.
        temperature: Sampling temperature.
        use_quality_filter: If True, filter to QED>=0.6 and SA<=4.

    Returns:
        (valid_x, log_rnd, scalar_rewards, sampling_trace)
    """
    xt, log_rnd, trace = _diffusion_loop(
        model, steps, mask, pad, batch_size, max_length,
        quality_mode=quality_mode,
        compute_rnd=True,
        pretrained=pretrained,
        remasking_mode=remasking_mode,
        num_remasking=num_remasking,
        quality_threshold=quality_threshold,
        temperature=temperature,
    )

    device = xt.device
    samples = xt.to(device)

    validSequences, valid_indices = _decode_and_validate(model, tokenizer, samples)

    valid_x_final = [samples[idx] for idx in valid_indices]
    valid_log_rnd = [log_rnd[idx] for idx in valid_indices]

    print("len valid sequences:", len(validSequences))

    if len(validSequences) == 0:
        print("[WARNING] No valid molecules generated in this batch")
        empty_x = torch.empty((0, max_length), dtype=torch.long, device=device)
        empty_log_rnd = torch.empty((0,), dtype=torch.float32, device=device)
        empty_rewards = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_x, empty_log_rnd, empty_rewards, trace

    # Compute multi-objective rewards
    score_vectors = reward_model(input_seqs=validSequences)
    scalar_rewards = np.sum(score_vectors, axis=-1)
    scalar_rewards = torch.as_tensor(scalar_rewards, dtype=torch.float32, device=device)

    print(f"scalar reward dim{len(scalar_rewards)}")
    valid_log_rnd = torch.stack(valid_log_rnd, dim=0)

    log_rnd = valid_log_rnd + (scalar_rewards / alpha)
    valid_x_final = torch.stack(valid_x_final, dim=0)

    # Optionally filter to only keep quality sequences (QED >= 0.6 and SA <= 4)
    if use_quality_filter:
        qed_scores = score_vectors[:, 0]
        if score_vectors.shape[1] > 1:
            sa_scores = score_vectors[:, 1]
        else:
            _oracle_sa = Oracle('sa')
            raw_sa = np.array(_oracle_sa(validSequences))
            sa_scores = raw_sa
        quality_mask = (qed_scores >= 0.6) & (sa_scores <= 4)

        n_quality = quality_mask.sum()
        print(f"Quality filtering: {n_quality}/{len(validSequences)} sequences pass (QED>=0.6, SA<=4)")

        if n_quality == 0:
            print("[WARNING] No quality molecules in this batch")
            empty_x = torch.empty((0, max_length), dtype=torch.long, device=device)
            empty_log_rnd = torch.empty((0,), dtype=torch.float32, device=device)
            empty_rewards = torch.empty((0,), dtype=torch.float32, device=device)
            return empty_x, empty_log_rnd, empty_rewards, trace

        quality_mask_torch = torch.as_tensor(quality_mask, dtype=torch.bool, device=device)

        quality_x_final = valid_x_final[quality_mask_torch]
        quality_log_rnd = log_rnd[quality_mask_torch]
        quality_rewards = scalar_rewards[quality_mask_torch]
    else:
        print(f"No quality filtering applied - using all {len(validSequences)} valid molecules")
        quality_x_final = valid_x_final
        quality_log_rnd = log_rnd
        quality_rewards = scalar_rewards

    return quality_x_final, quality_log_rnd, quality_rewards, trace


@torch.no_grad()
def sample_mol_eval(
    model, reward_model, tokenizer,
    steps, mask, pad, batch_size, max_length,
    quality_mode="both",
    remasking_mode="schedule_aware",
    num_remasking=1,
    quality_threshold=1,
    temperature=1.0,
    evaluator=None,
    dataframe=False,
    unmask_quality_threshold=None,
):
    """Generate molecules for evaluation.

    Args:
        model: Finetuned policy model.
        reward_model: Molecule scoring function.
        tokenizer: SAFE tokenizer for decoding.
        steps: Number of diffusion steps.
        mask: Mask token ID.
        pad: Pad token ID.
        batch_size: Number of sequences to generate.
        max_length: Maximum sequence length.
        quality_mode: "none", "both", "unmasking_only", or "insertion_only".
        remasking_mode: Remasking strategy.
        num_remasking: Number of tokens to remask per step.
        quality_threshold: Threshold for insertion quality filtering. Pass None
            to use schedule-driven deletion with no threshold gate
        temperature: Sampling temperature.
        evaluator: TDC Evaluator for diversity (created if None).
        dataframe: If True, include a pandas DataFrame in the return.

    Returns:
        Without dataframe:
            (validSequences, qed, sa, uniqueness, diversity, quality, valid_fraction)
        With dataframe:
            (validSequences, qed, sa, valid_fraction, uniqueness, diversity, quality, df)
        validSequences is the raw list including duplicates; qed/sa are scored
        on the unique set. Caller can dedup with set(validSequences). The
        dataframe (when requested) has one row per unique molecule.
    """
    if evaluator is None:
        evaluator = Evaluator('diversity')

    xt, _, trace = _diffusion_loop(
        model, steps, mask, pad, batch_size, max_length,
        quality_mode=quality_mode,
        compute_rnd=False,
        remasking_mode=remasking_mode,
        num_remasking=num_remasking,
        quality_threshold=quality_threshold,
        temperature=temperature,
        unmask_quality_threshold=unmask_quality_threshold,
    )

    device = xt.device
    samples = xt.to(device)

    decoded_samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

    use_bracket_safe = model.config.training.get('use_bracket_safe', False)
    smiles_samples = batch_safe_to_smiles(decoded_samples, use_bracket_safe=use_bracket_safe, fix=True)

    # Extract valid sequences (take largest fragment)
    validSequences = [sorted(s.split('.'), key=len)[-1] for s in smiles_samples if s]

    print("len valid sequences:", len(validSequences))
    valid_fraction = len(validSequences) / batch_size
    uniqueSequences = list(set(validSequences))
    uniqueness = len(uniqueSequences) / len(validSequences) if len(validSequences) > 0 else 0
    diversity = evaluator(uniqueSequences) if len(uniqueSequences) > 0 else 0

    # Calculate quality (unique sequences with QED >= 0.6 and SA <= 4)
    if len(uniqueSequences) > 0:
        score_vectors_temp = reward_model(input_seqs=list(uniqueSequences))
        qed_scores = score_vectors_temp[:, 0]  # Raw QED (0-1)

        # Always use raw SA (1-10 scale) for quality filtering
        _oracle_sa = Oracle('sa')
        raw_sa_scores = np.array(_oracle_sa(list(uniqueSequences)))

        quality_count = sum((qed_scores >= 0.6) & (raw_sa_scores <= 4))
        quality = quality_count / batch_size
        print(f'Quality:\t{quality}')

        qed = qed_scores
        sa = raw_sa_scores
    else:
        zeros = [0.0]
        qed = zeros
        sa = zeros
        quality = 0.0

    if dataframe:
        df = pd.DataFrame({
            "Mol Sequence": uniqueSequences,
            "QED": qed if len(uniqueSequences) else [0.0],
            "SA": sa if len(uniqueSequences) else [0.0],
        })
        return validSequences, qed, sa, valid_fraction, uniqueness, diversity, quality, df

    return validSequences, qed, sa, uniqueness, diversity, quality, valid_fraction
