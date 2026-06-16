import abc
from typing import Optional
import torch
from torch import Tensor
from dataclasses import dataclass
from .schedule import Schedule
import torch.nn.functional as F


@dataclass
class ModelPrediction:
    token_logits: Tensor
    length_posterior: Optional[Tensor]
    expected_gaps: Tensor

    def __init__(
        self,
        token_logits: Tensor,
        length_posterior: Optional[Tensor] = None,
        expected_gaps: Optional[Tensor] = None,
    ):
        assert length_posterior is not None or expected_gaps is not None
        self.token_logits = token_logits
        self.length_posterior = length_posterior
        self.expected_gaps = expected_gaps
        if self.expected_gaps is None:
            _, _, L = self.length_posterior.shape
            index = torch.arange(0, L, device=token_logits.device).view(1, 1, -1)
            self.expected_gaps = (F.softmax(self.length_posterior, dim=-1) * index).sum(dim=-1)


@dataclass
class Rate:
    unmask_rate: Tensor  # Shape [Batch, Length, Vocab]
    length_rate: Tensor  # Shape [Batch]


@dataclass
class HittingTime:
    insertion_time: Tensor  # Shape [Batch, Length]
    unmasking_time: Tensor  # Shape [Batch, Length]

    def __iter__(self):
        yield from [self.insertion_time, self.unmasking_time]


@dataclass
class JointInterpolantResult:
    # Joint Interpolant
    xt: Tensor  # Shape [Batch, Length]
    st: Tensor  # Shape [Batch, Length]
    _x1: Tensor
    _pad_token: int
    _mask_token: int

    @property
    def mask_indices(self) -> Tensor:
        return self.xt == self._mask_token

    @property
    def unmasked(self) -> Tensor:
        return torch.gather(self._x1, 1, self.st)

    @property
    def xt_length(self) -> Tensor:
        # Calculate length of xt
        return (self.xt != self._pad_token).sum(dim=1)

    @property
    def x1_length(self) -> Tensor:
        # Calculate length of x1
        return (self._x1 != self._pad_token).sum(dim=1)

    @property
    def gaps_and_mask(self) -> tuple[Tensor, Tensor]:
        x1_len = self.x1_length
        gaps = self.st.clone()

        pad_front = gaps.new_zeros((gaps.shape[0], 1)) - 1  # -1 for the front padding
        pad_back = gaps.new_zeros((gaps.shape[0], 1))
        gaps = torch.cat([pad_front, gaps, pad_back], dim=1)  # Add a leading zero

        gaps.scatter_(
            1, self.xt_length.unsqueeze(1) + 1, x1_len.unsqueeze(1)
        )  # Fill the last position with x1_len

        gaps = gaps[:, 1:] - gaps[:, :-1] - 1
        gaps = torch.clamp(gaps, min=0)

        idx = torch.arange(gaps.size(1), device=self.xt.device).unsqueeze(
            0
        )  # shape [1, max_gap]
        mask = idx <= self.xt_length.unsqueeze(1)
        gaps[~mask] = 0

        return gaps, mask


class JointInterpolant(abc.ABC):
    def __init__(
        self,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        """
        TODO: Add knobs
        """
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.vocab_size = vocab_size

    @abc.abstractmethod
    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        Shape:
            t: [B]
        Returns:
            weight_unmask: [B, L]
            weight_delete: [B, L+1]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_actual_rate(self, prediction: ModelPrediction, t: Tensor) -> Rate:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_interpolant(self, t: Tensor, x1: Tensor) -> JointInterpolantResult:
        """
        Sample the interpolant xt from x1 at time t
        Shapes:
            x1: [B, L]
            t: [B]
        Returns:
            xt: [B, L]
            st: [B, L] boolean mask of positions that corresponds to xt
            xt_mask_indices: [B, L] boolean mask of positions that are masked at xt
            x1_remained: [B, L] tokens that are not deleted, used for the training target
            gap_counts: [B, L+1] the number of deleted tokens between xt slots
        """
        raise NotImplementedError


class AnyOrderMaskInsertionInterpolant(JointInterpolant):
    def __init__(
        self,
        insertion_schedule: Schedule,
        unmask_schedule: Schedule,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        super().__init__(vocab_size, mask_token, pad_token, max_length)
        self.insertion_schedule = insertion_schedule
        self.unmask_schedule = unmask_schedule
        #self.max_length = 500

    def expected_mask_fraction(self, t: Tensor, xt: Tensor) -> Tensor:
        """
        Compute the expected fraction of tokens that should be masked at time t.
        For AnyOrderMaskInsertionInterpolant, tokens are:
        - Deleted (pad) if t < insertion_time  
        - Masked if insertion_time <= t < unmasking_time
        - Unmasked if t >= unmasking_time
        
        We approximate: E[fraction masked] ≈ max(0, insertion_schedule.at(t) - unmask_schedule.at(t))
        
        Args:
            t: [B] current time
            xt: [B, L] current sequence (to get current length)
        Returns:
            [B] expected number of masked tokens per sequence
        """
        # Get schedule values at time t
        insertion_progress = self.insertion_schedule.at(t)  # [B]
        unmask_progress = self.unmask_schedule.at(t)  # [B]
        
        # Expected fraction of tokens that are inserted but not yet unmasked
        # Clamp to ensure non-negative
        expected_mask_frac = torch.clamp(insertion_progress - unmask_progress, min=0.0, max=1.0)
        
        # Get current sequence length (non-pad tokens)
        current_length = (xt != self.pad_token).sum(dim=1).float()  # [B]
        
        # Expected number of masked tokens
        expected_num_masked = expected_mask_frac * current_length  # [B]
        
        return expected_num_masked

    def hitting_time(self, t: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        """
        t1 is sampled from a uniform distribution over [0, 1]. when t1 < self.mask_schedule.at(t)
        t2 is sampled from a uniform distribution over [t1, 1]
        """
        B, L = x1.shape
        eps = 1e-6

        insert_time = self.insertion_schedule.sample((B, L), device=x1.device)
        insert_time = eps + (1 - eps) * insert_time  # ensure t1 is not 0
        unmask_time = self.unmask_schedule.sample_truncated(
            insert_time, (B, L), device=x1.device
        )

        return insert_time, unmask_time

    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        """
        insert_weight = self.insertion_schedule.rate_scale_factor(t)
        insert_weight = insert_weight[:, None].expand(-1, x1.shape[1] + 1)

        unmask_weight = self.unmask_schedule.rate_scale_factor(t)
        unmask_weight = unmask_weight.unsqueeze(1).expand(-1, x1.shape[1])

        return unmask_weight, insert_weight

    def to_actual_rate(
        self, xt: Tensor, prediction: ModelPrediction, t: Tensor
    ) -> Rate:
        """
        Return the actual rate for the sampling
        Args:
            xt: [B, L] the sampled tokens
            prediction: ModelPrediction object containing token_posterior and expected_gaps
            t: [B] the time parameter
        """
        token_posterior = F.softmax(prediction.token_logits, dim=-1)  # (B, L, V)
        unmask_rate = token_posterior * self.unmask_schedule.rate_scale_factor(t).view(
            -1, 1, 1
        )
        
        length_rate = (
            prediction.expected_gaps
            * self.insertion_schedule.rate_scale_factor(t).view(-1, 1)
        )
        #print("expected_gaps:", prediction.expected_gaps, "length_rate:", length_rate)

        return Rate(
            unmask_rate=unmask_rate,  # (B, L, V)
            length_rate=length_rate,  # (B, L+1)
        )

    def sample_interpolant(self, t: Tensor, x1: Tensor) -> JointInterpolantResult:
        """
        Shapes:
            x1: [B, L]
            t: [B]
        Returns:
            xt: [B, L]
            st: [B, L] boolean mask of positions that corresponds to xt
            xt_mask_indices: [B, L] boolean mask of positions that are masked at xt
            x1_remained: [B, L] tokens that are not deleted, used for the training target
            gap_counts: [B, L+1] the number of deleted tokens between xt slots
        """
        # sample the stopping time (B, L, 2)
        insertion_time, unmasking_time = self.hitting_time(t, x1)

        clean_tokens = x1.ne(self.pad_token)
        deleted_tokens = clean_tokens & (t[:, None] < insertion_time)
        masked_tokens = (
            clean_tokens
            & (t[:, None] >= insertion_time)
            & (t[:, None] < unmasking_time)
        )

        xt = torch.where(
            deleted_tokens,
            self.pad_token,  # for deletion, change to pad token
            torch.where(
                masked_tokens,
                self.mask_token,  # for masking, change to mask token
                x1,
            ),
        )

        st = xt.ne(self.pad_token).to(torch.int32).argsort(dim=1, descending=True, stable=True) # edited to sort integers
        xt = torch.gather(xt, 1, st)
        st[xt == self.pad_token] = 0

        return JointInterpolantResult(
            xt=xt, st=st, _x1=x1, _pad_token=self.pad_token, _mask_token=self.mask_token
        )
        
    def sample_interpolant_plan(self, t: Tensor, x1: Tensor) -> JointInterpolantResult:
        """
        Shapes:
            x1: [B, L]
            t: [B]
        Returns:
            xt: [B, L]
            st: [B, L] boolean mask of positions that corresponds to xt
            xt_mask_indices: [B, L] boolean mask of positions that are masked at xt
            x1_remained: [B, L] tokens that are not deleted, used for the training target
            gap_counts: [B, L+1] the number of deleted tokens between xt slots
        """
        # sample the stopping time (B, L, 2)
        insertion_time, unmasking_time = self.hitting_time(t, x1)

        clean_tokens = x1.ne(self.pad_token)
        deleted_tokens = clean_tokens & (t[:, None] < insertion_time)
        masked_tokens = (
            clean_tokens
            & (t[:, None] >= insertion_time)
            & (t[:, None] < unmasking_time)
        )

        xt = torch.where(
            deleted_tokens,
            self.pad_token,  # for deletion, change to pad token
            torch.where(
                masked_tokens,
                self.mask_token,  # for masking, change to mask token
                x1,
            ),
        )
        st = xt.ne(self.pad_token).to(torch.int32).argsort(dim=1, descending=True, stable=True) # edited to sort integers
        xt = torch.gather(xt, 1, st)
        st[xt == self.pad_token] = 0
        num_gaps = (st != 0).sum(dim=1) + 1  # [B]

        deleted_mask = deleted_tokens  # [B, L]
        
        # Create gap assignment tensor: gap_assignment[b, gap_idx, x1_pos] = 1 if x1_pos is in gap gap_idx
        B, L = x1.shape
        max_gaps = L + 1
        gap_assignment = torch.zeros(B, max_gaps, L, device=x1.device, dtype=torch.float)
        
        # For each deleted position in x1, determine which gap it belongs to
        # Gap index = number of non-deleted positions (st values) that come before it
        pos_indices = torch.arange(L, device=x1.device).view(1, L, 1)  # [1, L, 1]
        st_expanded = st.unsqueeze(1)  # [B, 1, L]
        st_valid_mask = (st != 0).unsqueeze(1)  # [B, 1, L]
        
        # Count how many valid st entries are less than each position
        # gap_indices[b, pos] = number of st values < pos for deleted positions
        gap_indices = ((st_expanded < pos_indices) & st_valid_mask).sum(dim=2)  # [B, L]
        
        # Set gap_assignment[b, gap_idx, pos] = 1 where pos is deleted and belongs to gap_idx
        batch_idx = torch.arange(B, device=x1.device).view(B, 1).expand(B, L)
        pos_idx = torch.arange(L, device=x1.device).view(1, L).expand(B, L)
        
        gap_assignment[batch_idx[deleted_mask], gap_indices[deleted_mask], pos_idx[deleted_mask]] = 1.0

        return JointInterpolantResult(
            xt=xt, st=st, _x1=x1, _pad_token=self.pad_token, _mask_token=self.mask_token
        ), deleted_mask, gap_assignment


class MDMInterpolant(JointInterpolant):
    def __init__(
        self,
        unmask_schedule: Schedule,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        super().__init__(vocab_size, mask_token, pad_token, max_length)
        self.unmask_schedule = unmask_schedule

    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        there's no weight_delete for the vanilla MDM
        """
        weight_unmask = self.unmask_schedule.rate_scale_factor(t)
        weight_unmask_expanded = weight_unmask.unsqueeze(1).expand(
            -1, x1.shape[1]
        )  # (B,L)
        return weight_unmask_expanded

    def to_actual_rate(self, xt: Tensor, prediction: Tensor, t: Tensor) -> Rate:
        """
        Return the actual rate for the sampling
        """
        token_posterior = F.softmax(prediction, dim=-1)  # (B, L, V)
        unmask_rate = token_posterior * self.unmask_schedule.rate_scale_factor(t).view(
            -1, 1, 1
        )

        return Rate(
            unmask_rate=unmask_rate,  # (B, L, V)
            length_rate=None,  # (B, L+1)
        )

    def sample_interpolant(self, t: Tensor, x1: Tensor) -> JointInterpolantResult:
        # sample the stopping time (B, L, 2)
        eps = 1e-6
        unmask_time = self.unmask_schedule.sample(
            (x1.shape[0], x1.shape[1]), device=x1.device
        )
        unmask_time = unmask_time * (1 - eps) + eps

        xt = torch.where(
            t[:, None] < unmask_time,
            self.mask_token,  # for masking, change to mask token
            x1,
        )
        st = torch.arange(xt.shape[1], device=xt.device, dtype=torch.long).repeat(
            xt.shape[0], 1
        )

        return JointInterpolantResult(
            xt=xt, st=st, _x1=x1, _pad_token=self.pad_token, _mask_token=self.mask_token
        )
