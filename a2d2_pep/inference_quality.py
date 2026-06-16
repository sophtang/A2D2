"""Unified peptide sampling with quality-guided planning.

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

import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sampling import SamplingResult, SamplingTraceDatapoint, _sample_tokens
from remasking_scheduleaware import apply_schedule_aware_remasking, apply_schedule_aware_insertion

QUALITY_MODES = {"none", "both", "unmasking_only", "insertion_only"}

# When set (e.g. A2D2_QUALITY_DEBUG=1), the diffusion loop prints, per step, how
# many already-unmasked tokens get remasked and how many proposed insertions get
# filtered by the quality planner, plus a per-batch total. Off by default so it
# never spams training/eval runs.
_QUALITY_DEBUG = os.environ.get("A2D2_QUALITY_DEBUG", "") not in ("", "0", "false", "False")


@torch.no_grad()
def _diffusion_loop(
    model, steps, mask, pad, batch_size, max_length,
    quality_mode="both",
    compute_rnd=False,
    pretrained=None,
    remasking_mode="schedule_aware",
    num_remasking=1,
    quality_threshold=1,
    unmask_quality_threshold=None,
    unmask_all=False,
    freq_penalty=0.0,
    return_trace=False,
):
    """Core discrete diffusion sampling loop for peptide generation.

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

    dbg_total_remasked = 0
    dbg_total_proposed_ins = 0
    dbg_total_filtered = 0

    for i in range(steps):
        step_remasked = 0
        step_proposed_ins = 0
        step_filtered = 0
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

        # Remove mask token from sampling so every masked position is decoded.
        # The final step always does this; unmask_all does it every step, so the
        # schedule-aware remasking below re-masks the lowest-quality tokens back
        # down to the schedule's expected mask count.
        if i == steps - 1 or unmask_all:
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

        # --- Frequency penalty: down-weight residues already abundant in the
        # sequence so (re)decoded masked positions don't collapse onto the modal
        # token (glycine). Only masked positions are sampled; clean positions are
        # overwritten below, so penalizing the whole tensor is harmless. mask/pad
        # never accumulate counts, so their entries stay untouched. Applied to a
        # copy so trans_prob (used for RND log-probs) is unchanged.
        sample_prob = trans_prob
        if freq_penalty > 0.0:
            V = trans_prob.shape[-1]
            clean_tok = (xt != mask) & (xt != pad)  # (B, L)
            counts = torch.zeros(batch_size, V, device=device, dtype=trans_prob.dtype)
            counts.scatter_add_(1, torch.where(clean_tok, xt, torch.zeros_like(xt)),
                                clean_tok.to(trans_prob.dtype))
            sample_prob = trans_prob * torch.exp(-freq_penalty * counts).unsqueeze(1)

        new_xt = _sample_tokens(sample_prob)
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

            if remasking_mode == "remdm":
                remasking_score_temp = torch.rand(remasking_conf.shape, device=device)
            elif remasking_mode == "remdm_conf":
                remasking_score_temp = -1.0 * remasking_conf
            elif remasking_mode == "schedule_aware":
                # Only remask when the unmasking planner is active. Otherwise
                # (e.g. insertion_only / no_unmasking_planner) remasking_conf is
                # all zeros, so this would remask schedule-excess tokens by
                # position rather than by quality.
                if not disable_unmasking_planner:
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

            if _QUALITY_DEBUG:
                # Positions that were clean before this remasking block and are
                # now mask are exactly the unmasked tokens that got remasked.
                step_remasked = int((clean_index & (new_xt == mask)).sum().item())

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

            if _QUALITY_DEBUG:
                # ext has been masked by the max-length validity check above, so
                # this is the number of fresh mask tokens actually inserted.
                step_proposed_ins = int(ext.sum().item())

            # Schedule-aware insertion quality filtering
            if use_remasking and not disable_insertion_planner:
                if compute_rnd:
                    xt_tmp_before = xt_tmp.clone()

                dbg_nonpad_before = int((xt_tmp != pad).sum().item()) if _QUALITY_DEBUG else 0

                xt_tmp = apply_schedule_aware_insertion(
                    model, xt_tmp, new_xt, t, dt, ext, mask, pad, max_length,
                    orig_mask, new_pos_orig, quality_threshold
                )

                if _QUALITY_DEBUG:
                    # Filtering only drops/compacts tokens, so the drop in
                    # non-pad count is the number of insertions filtered out.
                    step_filtered = dbg_nonpad_before - int((xt_tmp != pad).sum().item())

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

        if _QUALITY_DEBUG:
            dbg_total_remasked += step_remasked
            dbg_total_proposed_ins += step_proposed_ins
            dbg_total_filtered += step_filtered
            print(
                f"[QUALITY {quality_mode}] step {i+1}/{steps}: "
                f"remasked {step_remasked} unmasked tokens -> mask | "
                f"insertions proposed {step_proposed_ins}, "
                f"filtered {step_filtered}, kept {step_proposed_ins - step_filtered}"
            )

        xt = xt_tmp
        t = t + dt

    if _QUALITY_DEBUG:
        print(
            f"[QUALITY {quality_mode}] TOTAL over {steps} steps (batch_size={batch_size}): "
            f"remasked {dbg_total_remasked} unmasked tokens | "
            f"insertions proposed {dbg_total_proposed_ins}, "
            f"filtered {dbg_total_filtered}, kept {dbg_total_proposed_ins - dbg_total_filtered}"
        )

    return xt, log_rnd, sampling_trace


@torch.no_grad()
def sample_peptides_buffer(
    model, reward_model, analyzer, tokenizer,
    steps, mask, pad, batch_size, max_length,
    quality_mode="both",
    compute_rnd=False,
    pretrained=None,
    alpha=0.1,
    remasking_mode="schedule_aware",
    num_remasking=1,
    quality_threshold=1,
    min_length=0,
):
    """Generate peptides for training buffer.

    Args:
        model: Finetuned policy model.
        reward_model: Multi-objective scoring function.
        analyzer: PeptideAnalyzer for validation.
        tokenizer: Tokenizer for decoding.
        steps: Number of diffusion steps.
        mask: Mask token ID.
        pad: Pad token ID.
        batch_size: Number of sequences to generate.
        max_length: Maximum sequence length.
        quality_mode: "none", "both", "unmasking_only", or "insertion_only".
        compute_rnd: If True, compute step-wise log importance weights (requires pretrained).
                     If False, returns placeholder zero log_rnd (for ELBO-based RND).
        pretrained: Frozen pretrained model (required when compute_rnd=True).
        alpha: RND scaling factor.
        remasking_mode: Remasking strategy.
        num_remasking: Number of tokens to remask per step.
        quality_threshold: Threshold for insertion quality filtering.

    Returns:
        (valid_x, log_rnd, scalar_rewards, sampling_trace)
    """
    xt, log_rnd, trace = _diffusion_loop(
        model, steps, mask, pad, batch_size, max_length,
        quality_mode=quality_mode,
        compute_rnd=compute_rnd,
        pretrained=pretrained,
        remasking_mode=remasking_mode,
        num_remasking=num_remasking,
        quality_threshold=quality_threshold,
    )

    device = xt.device
    decoded_samples = tokenizer.batch_decode(xt)

    valid_x_final = []
    validSequences = []
    valid_log_rnd = []

    for idx, seq in enumerate(decoded_samples):
        if not analyzer.is_peptide(seq):
            continue
        token_len = int((xt[idx] != pad).sum().item())
        if min_length > 0 and token_len < min_length:
            continue
        valid_x_final.append(xt[idx])
        validSequences.append(seq)
        if compute_rnd:
            valid_log_rnd.append(log_rnd[idx])

    print("len valid sequences:", len(validSequences))

    if len(validSequences) == 0:
        print("[WARNING] No valid peptides generated in this batch")
        empty_x = torch.empty((0, max_length), dtype=torch.long, device=device)
        empty_log_rnd = torch.empty((0,), dtype=torch.float32, device=device)
        empty_rewards = torch.empty((0,), dtype=torch.float32, device=device)
        return empty_x, empty_log_rnd, empty_rewards, trace

    score_vectors = reward_model(input_seqs=validSequences)
    scalar_rewards = np.sum(score_vectors, axis=-1)
    scalar_rewards = torch.as_tensor(scalar_rewards, dtype=torch.float32, device=device)

    print(f"scalar reward dim{len(scalar_rewards)}")
    valid_x_final = torch.stack(valid_x_final, dim=0)

    if compute_rnd:
        valid_log_rnd = torch.stack(valid_log_rnd, dim=0)
        log_rnd_out = valid_log_rnd + (scalar_rewards / alpha)
    else:
        log_rnd_out = torch.zeros(len(validSequences), dtype=torch.float32, device=device)

    return valid_x_final, log_rnd_out, scalar_rewards, trace


@torch.no_grad()
def sample_peptides_eval(
    model, reward_model, analyzer, tokenizer,
    steps, mask, pad, batch_size, max_length,
    quality_mode="both",
    remasking_mode="schedule_aware",
    num_remasking=1,
    quality_threshold=1,
    unmask_quality_threshold=None,
    unmask_all=False,
    freq_penalty=0.0,
    dataframe=False,
    return_valid=False,
):
    """Generate peptides for evaluation.

    Args:
        model: Finetuned policy model.
        reward_model: Multi-objective scoring function.
        analyzer: PeptideAnalyzer for validation.
        tokenizer: Tokenizer for decoding.
        steps: Number of diffusion steps.
        mask: Mask token ID.
        pad: Pad token ID.
        batch_size: Number of sequences to generate.
        max_length: Maximum sequence length.
        quality_mode: "none", "both", "unmasking_only", or "insertion_only".
        remasking_mode: Remasking strategy.
        num_remasking: Number of tokens to remask per step.
        quality_threshold: Threshold for insertion quality filtering. 
        dataframe: If True, include a pandas DataFrame in the return.
        return_valid: If True, return decoded valid sequences instead of raw token tensors.

    Returns:
        For multi-objective (5 objectives):
            (samples, affinity, sol, hemo, nf, permeability, valid_fraction[, df])
        For single objective:
            (samples, sol, valid_fraction[, df])
        When return_valid=True, samples is replaced with validSequences list.
    """
    xt, _, trace = _diffusion_loop(
        model, steps, mask, pad, batch_size, max_length,
        quality_mode=quality_mode,
        compute_rnd=False,
        remasking_mode=remasking_mode,
        num_remasking=num_remasking,
        quality_threshold=quality_threshold,
        unmask_quality_threshold=unmask_quality_threshold,
        unmask_all=unmask_all,
        freq_penalty=freq_penalty,
    )

    device = xt.device
    samples = xt.to(device)
    decoded_samples = tokenizer.batch_decode(samples)

    valid_x_final = []
    validSequences = []

    for idx, seq in enumerate(decoded_samples):
        if analyzer.is_peptide(seq):
            valid_x_final.append(samples[idx])
            validSequences.append(seq)

    print("len valid sequences:", len(validSequences))

    valid_fraction = len(validSequences) / batch_size

    # Determine number of objectives from reward model
    num_objectives = len(reward_model.score_func_names) if hasattr(reward_model, 'score_func_names') else 5

    if len(validSequences) != 0:
        score_vectors = reward_model(input_seqs=validSequences)  # (N, num_objectives)
        average_scores = score_vectors.T

        if num_objectives == 1:
            sol = average_scores[0]
        else:
            affinity = average_scores[0]
            sol = average_scores[1]
            hemo = average_scores[2]
            nf = average_scores[3]
            permeability = average_scores[4]
    else:
        zeros = [0.0]

        if num_objectives == 1:
            sol = zeros
        else:
            affinity = zeros
            sol = zeros
            hemo = zeros
            nf = zeros
            permeability = zeros

    if num_objectives == 1:
        if dataframe:
            df = pd.DataFrame({
                "Peptide Sequence": validSequences,
                "Solubility": sol if len(validSequences) else [0.0],
            })
            if return_valid:
                return validSequences, sol, valid_fraction, df
            return samples, sol, valid_fraction, df

        if return_valid:
            return validSequences, sol, valid_fraction
        return samples, sol, valid_fraction

    if dataframe:
        df = pd.DataFrame({
            "Peptide Sequence": validSequences,
            "Binding Affinity": affinity if len(validSequences) else [0.0],
            "Solubility": sol if len(validSequences) else [0.0],
            "Hemolysis": hemo if len(validSequences) else [0.0],
            "Nonfouling": nf if len(validSequences) else [0.0],
            "Permeability": permeability if len(validSequences) else [0.0],
        })
        if return_valid:
            return validSequences, affinity, sol, hemo, nf, permeability, valid_fraction, df
        return samples, affinity, sol, hemo, nf, permeability, valid_fraction, df

    if return_valid:
        return validSequences, affinity, sol, hemo, nf, permeability, valid_fraction
    return samples, affinity, sol, hemo, nf, permeability, valid_fraction
