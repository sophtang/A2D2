"""
Schedule-aware remasking and insertion logic that ensures the number of masked tokens
follows the interpolant schedule.
"""
import torch
import numpy as np

def apply_schedule_aware_insertion(
    model,
    xt_tmp,
    new_xt,
    t,
    dt,
    ext,
    mask,
    pad,
    max_length,
    orig_mask,
    new_pos_orig,
    quality_threshold=1,
):
    """
    Remove low-quality insertions based on insertion confidence while respecting
    the interpolant schedule for expected sequence length.
    
    Args:
        model: Model with planner and interpolant
        xt_tmp: Sequence after insertion [B, L]
        new_xt: Sequence before insertion [B, L]
        t: Current time [B]
        dt: Time step size
        ext: Number of insertions per gap [B, L+1]
        mask: Mask token ID
        pad: Pad token ID
        max_length: Maximum sequence length
        orig_mask: Mask of original token positions [B, L]
        new_pos_orig: New positions of original tokens [B, L]
        quality_threshold: If a float, drop insertions with confidence below it; if None, use schedule-driven deletion
        
    Returns:
        xt_tmp: Modified sequence with low-quality insertions removed (respecting schedule)
    """
    device = xt_tmp.device
    batch_size, L = xt_tmp.shape
    total_ext = ext.sum(dim=1)

    # Only proceed if there were insertions
    if total_ext.sum() == 0:
        return xt_tmp

    # Get planner predictions on inserted state. The insertion head is trained
    # with the pre-step time t (see loss_insert_planner_flexible), so condition
    # on t here too; t_next is still used below for the length schedule.
    t_next = t + dt
    planner_out = model.planner(xt_tmp, t)
    insertion_conf = planner_out.get("insertion_conf", None)

    if insertion_conf is None:
        return xt_tmp

    insertion_conf = insertion_conf.squeeze(-1)  # (B, L)

    # Expected sequence length at next timestep according to schedule
    current_length_after = xt_tmp.ne(pad).sum(dim=1).float()  # [B]
    expected_progress = model.interpolant.insertion_schedule.at(t_next)  # [B]
    estimated_final_length = current_length_after / (expected_progress.clamp(min=0.1))
    expected_length = estimated_final_length * expected_progress  # [B]

    # Mark positions in xt_tmp that came from new_xt (originals) vs. fresh insertions.
    # Fancy-indexing scatter avoids the per-batch python loop.
    valid_b, valid_l = orig_mask.nonzero(as_tuple=True)
    valid_p = new_pos_orig[valid_b, valid_l].long().clamp_(0, L - 1)
    is_original = torch.zeros_like(xt_tmp, dtype=torch.bool)
    is_original[valid_b, valid_p] = True
    inserted_positions = (xt_tmp == mask) & ~is_original

    # Two deletion modes, selected by `quality_threshold`:
    #   * float: drop insertions whose confidence is below the threshold, capped
    #     so the length never falls below the scheduled minimum.
    candidates = inserted_positions & (insertion_conf < quality_threshold)
    num_bad = candidates.sum(dim=1)  # [B], long
    min_length = expected_length.long().clamp(min=1)  # [B]
    max_removable = (current_length_after.long() - min_length).clamp(min=0)
    length_after_removal = current_length_after.long() - num_bad
    schedule_violates = length_after_removal < min_length
    k_per_row = torch.where(schedule_violates, max_removable, num_bad)
    k_per_row = torch.where(num_bad > 0, k_per_row, torch.zeros_like(k_per_row))

    if not candidates.any():
        return xt_tmp

    # Select the lowest-confidence candidates per row via a sort.
    neg_inf = torch.tensor(float('-inf'), device=device, dtype=insertion_conf.dtype)
    scores = torch.where(candidates, -insertion_conf, neg_inf)  # higher = worse
    _, sorted_indices = scores.sort(dim=1, descending=True)
    positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
    keep_in_topk = positions < k_per_row.unsqueeze(1)  # [B, L]
    final_bad = torch.zeros_like(candidates)
    final_bad.scatter_(1, sorted_indices, keep_in_topk)

    if not final_bad.any():
        return xt_tmp

    # Compact each row to the left (keep good, drop bad), then pad the tail.
    # Stable sort by the bad flag pushes bad positions to the right.
    sort_key = final_bad.long()
    _, perm = torch.sort(sort_key, dim=1, stable=True)
    xt_tmp = torch.gather(xt_tmp, 1, perm)
    num_keep = (~final_bad).sum(dim=1)  # [B]
    tail_mask = positions >= num_keep.unsqueeze(1)  # [B, L]
    xt_tmp = torch.where(tail_mask, torch.full_like(xt_tmp, pad), xt_tmp)

    return xt_tmp


def apply_schedule_aware_remasking(
    model,
    new_xt,
    t,
    dt,
    remasking_conf,
    clean_index,
    mask,
    neg_inf,
    batch_size,
    unmask_quality_threshold=None,
):
    """
    Apply schedule-aware remasking: adjust number of masks to match expected count from schedule.

    Args:
        model: Model with interpolant that has an unmask_schedule
        new_xt: Current sequence [B, L]
        t: Current time [B]
        dt: Time step size
        remasking_conf: Confidence scores for tokens [B, L]
        clean_index: Boolean mask of clean tokens (not mask, not pad) [B, L]
        mask: Mask token ID
        neg_inf: Negative infinity tensor
        batch_size: Batch size
        unmask_quality_threshold: If None (default), remask exactly the schedule
            excess (count-based). If a float, ignore the schedule budget entirely
            and remask EVERY clean token whose unmasking-quality confidence is
            below the threshold. Higher threshold => more aggressive remasking.

    Returns:
        new_xt: Modified sequence with schedule-aware remasking applied
    """
    # Threshold gate (overrides the schedule-driven count when set): remask every
    # clean token whose unmasking-quality confidence is below the threshold,
    # regardless of the schedule budget. Higher threshold => more remasking.
    if unmask_quality_threshold is not None:
        to_mask = clean_index & (remasking_conf < unmask_quality_threshold)
        return torch.where(to_mask, torch.full_like(new_xt, mask), new_xt)

    t_next = t + dt
    num_clean = clean_index.sum(dim=1)  # [B], long
    current_seq_len = (num_clean + (new_xt == mask).sum(dim=1)).float()  # [B]
    expected_unmasked_frac = model.interpolant.unmask_schedule.at(t_next)  # [B]
    expected_num_clean = expected_unmasked_frac * current_seq_len  # [B]
    masks_to_add = (num_clean.float() - expected_num_clean).round().long()  # [B]

    # Per-row k = min(masks_to_add, num_clean), clamped to >= 0.
    k_per_row = torch.minimum(masks_to_add.clamp(min=0), num_clean)  # [B]

    if k_per_row.sum() == 0:
        return new_xt

    # Use confidence to decide which clean tokens to remask: lowest conf first.
    remasking_score_temp = -1.0 * remasking_conf  # low conf = high score
    remasking_score_temp = torch.where(clean_index, remasking_score_temp, neg_inf)

    _, sorted_indices = remasking_score_temp.sort(dim=1, descending=True)
    L = remasking_score_temp.shape[1]
    positions = torch.arange(L, device=new_xt.device).unsqueeze(0)  # [1, L]
    keep_in_topk = positions < k_per_row.unsqueeze(1)  # [B, L]
    to_mask = torch.zeros_like(clean_index)
    to_mask.scatter_(1, sorted_indices, keep_in_topk)
    new_xt = torch.where(to_mask, torch.full_like(new_xt, mask), new_xt)

    return new_xt
