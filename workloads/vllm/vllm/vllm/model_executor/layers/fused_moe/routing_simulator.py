# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Token-to-Expert Routing Simulator

This module provides a framework for simulating and testing different
token-to-expert routing strategies for Mixture of Experts (MoE) models.
It supports routing logic customization and includes example implementations
like uniform random routing.
"""

import os
from abc import ABC, abstractmethod
import threading
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class RoutingStrategy(ABC):
    """Base class for token-to-expert routing strategies."""

    @abstractmethod
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        pass


class DistributionBasedRouting(RoutingStrategy):
    """
    Distribution-based random routing strategy with configurable distributions.

    This routing strategy randomly selects experts for each token based on
    different probability distributions. Currently supports uniform and normal
    distributions for testing different routing patterns.
    """

    def __init__(self, distribution: str = "uniform", **distribution_params: Any):
        """
        Initialize distribution-based routing.

        Args:
            distribution: Type of distribution to use for sampling
                - "uniform": Uniform distribution (default)
                - "normal": Normal/Gaussian distribution
            **distribution_params: Parameters specific to the
                chosen distribution
                For "uniform": No additional parameters needed
                For "normal": mean (default: 0.0), std (default: 1.0)
        """
        self.distribution = distribution.lower()
        self.distribution_params = distribution_params

        # Validate distribution and parameters
        self._validate_distribution_params()

    def _validate_distribution_params(self):
        """Validate distribution type and parameters."""
        valid_distributions = ["uniform", "normal"]

        if self.distribution not in valid_distributions:
            raise ValueError(
                f"Unsupported distribution: {self.distribution}. "
                f"Supported distributions: {valid_distributions}"
            )

        # Set default parameters if not provided
        if self.distribution == "normal":
            self.distribution_params.setdefault("mean", 0.0)
            self.distribution_params.setdefault("std", 1.0)

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select experts for each token using the specified distribution.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids) where:
            - topk_weights: Weights based on distribution sampling
            - topk_ids: Expert indices sampled from the distribution
        """
        num_tokens = hidden_states.shape[0]
        num_experts = router_logits.shape[-1]

        if indices_type is None:
            indices_type = torch.long

        # Generate expert IDs based on the specified distribution
        topk_ids = self._sample_expert_ids(
            num_tokens, num_experts, top_k, hidden_states.device, indices_type
        )

        # Generate weights based on the distribution
        topk_weights = self._generate_weights(num_tokens, top_k, hidden_states.device)

        return topk_weights, topk_ids

    def _sample_expert_ids(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        indices_type: torch.dtype,
    ) -> torch.Tensor:
        """Sample expert IDs based on the specified distribution."""

        if self.distribution == "uniform":
            # Uniform random sampling
            return torch.randint(
                low=0,
                high=num_experts,
                size=(num_tokens, top_k),
                dtype=indices_type,
                device=device,
            )

        elif self.distribution == "normal":
            # For normal distribution, sample continuous values and map to
            # expert IDs
            continuous_samples = self._sample_continuous_distribution(
                num_tokens, top_k, device
            )

            # Map continuous samples to expert indices
            # Normalize to [0, 1] range and scale to [0, num_experts)
            normalized_samples = self._normalize_samples(continuous_samples)
            expert_ids = (normalized_samples * num_experts).long()
            expert_ids = torch.clamp(expert_ids, 0, num_experts - 1)

            return expert_ids.to(dtype=indices_type)

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def _sample_continuous_distribution(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        """Sample from continuous distributions."""
        shape = (num_tokens, top_k)

        if self.distribution == "normal":
            mean = self.distribution_params["mean"]
            std = self.distribution_params["std"]
            return torch.normal(mean, std, size=shape, device=device)

        else:
            raise ValueError(
                f"Unsupported continuous distribution: {self.distribution}"
            )

    def _normalize_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Normalize samples to [0, 1] range."""
        if self.distribution == "normal":
            # Use sigmoid to map normal distribution to [0, 1]
            return torch.sigmoid(samples)

        else:
            raise ValueError(
                f"Unsupported distribution for normalization: {self.distribution}"
            )

    def _generate_weights(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        """Generate weights based on the distribution."""
        if self.distribution == "uniform":
            # All-ones weights for uniform distribution
            return torch.ones(
                (num_tokens, top_k),
                dtype=torch.float32,
                device=device,
            )

        elif self.distribution == "normal":
            # For normal distribution, generate weights from the same
            # distribution
            continuous_weights = self._sample_continuous_distribution(
                num_tokens, top_k, device
            )
            # Normalize to positive values and sum to 1
            weights = torch.abs(continuous_weights)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            return weights

        else:
            raise ValueError(
                f"Unsupported distribution for weight generation: {self.distribution}"
            )

    def get_distribution_info(self) -> dict:
        """Get information about the current distribution configuration."""
        return {
            "distribution": self.distribution,
            "parameters": self.distribution_params.copy(),
        }


class FixedExpertsRouting(RoutingStrategy):
    """Route every token to a fixed expert-id list for controlled experiments."""

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        num_experts = router_logits.shape[-1]
        if indices_type is None:
            indices_type = torch.long

        raw_ids = os.environ.get("VLLM_MOE_ROUTING_FIXED_EXPERT_IDS", "")
        expert_ids: list[int] = []
        for item in raw_ids.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            expert_id = int(item)
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError(
                    "VLLM_MOE_ROUTING_FIXED_EXPERT_IDS contains out-of-range "
                    f"expert_id={expert_id}; num_experts={num_experts}"
                )
            expert_ids.append(expert_id)
        if not expert_ids:
            raise ValueError(
                "VLLM_MOE_ROUTING_FIXED_EXPERT_IDS must contain at least one "
                "expert id when using fixed_experts routing."
            )

        while len(expert_ids) < top_k:
            expert_ids.extend(expert_ids)
        expert_ids = expert_ids[:top_k]

        ids = torch.tensor(expert_ids, dtype=indices_type, device=hidden_states.device)
        topk_ids = ids.unsqueeze(0).expand(num_tokens, top_k).contiguous()
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / max(1, top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        return topk_weights, topk_ids


class StaticTailHitRouting(RoutingStrategy):
    """Mostly static routing with a controlled offloaded-tail hit rate."""

    _counter_lock = threading.Lock()
    _assignment_counter = 0

    @staticmethod
    def _parse_expert_ids(raw: str) -> list[int]:
        expert_ids: list[int] = []
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            expert_ids.append(int(item))
        return expert_ids

    @classmethod
    def _next_assignment_offset(cls, count: int) -> int:
        with cls._counter_lock:
            offset = cls._assignment_counter
            cls._assignment_counter += count
            return offset

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        num_experts = router_logits.shape[-1]
        device = hidden_states.device
        if indices_type is None:
            indices_type = torch.long

        tail_ids = self._parse_expert_ids(
            os.environ.get("VLLM_MOE_ROUTING_STATIC_TAIL_EXPERT_IDS", "")
        )
        if not tail_ids:
            tail_count = int(os.environ.get("VLLM_MOE_ROUTING_STATIC_TAIL_COUNT", "5"))
            tail_start = max(0, num_experts - max(0, tail_count))
            tail_ids = list(range(tail_start, num_experts))
        for expert_id in tail_ids:
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError(
                    "VLLM_MOE_ROUTING_STATIC_TAIL_EXPERT_IDS contains "
                    f"out-of-range expert_id={expert_id}; "
                    f"num_experts={num_experts}"
                )

        base_ids = self._parse_expert_ids(
            os.environ.get("VLLM_MOE_ROUTING_STATIC_BASE_EXPERT_IDS", "")
        )
        if not base_ids:
            tail_id_set = set(tail_ids)
            base_ids = [
                expert_id
                for expert_id in range(num_experts)
                if expert_id not in tail_id_set
            ][:top_k]
        if not base_ids:
            raise ValueError("static_tail_hit routing has no base experts")
        for expert_id in base_ids:
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError(
                    "VLLM_MOE_ROUTING_STATIC_BASE_EXPERT_IDS contains "
                    f"out-of-range expert_id={expert_id}; "
                    f"num_experts={num_experts}"
                )

        while len(base_ids) < top_k:
            base_ids.extend(base_ids)
        base_ids = base_ids[:top_k]

        ids = torch.tensor(base_ids, dtype=indices_type, device=device)
        topk_ids = ids.unsqueeze(0).expand(num_tokens, top_k).clone()

        hit_probability = float(
            os.environ.get("VLLM_MOE_ROUTING_STATIC_TAIL_HIT_PROB", "0")
        )
        if hit_probability < 0.0 or hit_probability > 1.0:
            raise ValueError(
                "VLLM_MOE_ROUTING_STATIC_TAIL_HIT_PROB must be in [0, 1], "
                f"got {hit_probability}"
            )

        total_assignments = num_tokens * top_k
        if hit_probability > 0.0 and total_assignments > 0:
            cycle = 10000
            threshold = int(round(hit_probability * cycle))
            seed = int(os.environ.get("VLLM_MOE_ROUTING_STATIC_SEED", "0"))
            offset = self._next_assignment_offset(total_assignments)
            assignment_ids = (
                torch.arange(total_assignments, device=device, dtype=torch.long)
                + offset
            )
            hashed = (
                assignment_ids * 1103515245 + 12345 + seed
            ).remainder(cycle)
            hit_mask = hashed < threshold
            if torch.any(hit_mask):
                tail_tensor = torch.tensor(tail_ids, dtype=indices_type, device=device)
                tail_indices = hashed.remainder(len(tail_ids)).to(dtype=torch.long)
                flat_ids = topk_ids.reshape(-1)
                flat_ids[hit_mask] = tail_tensor[tail_indices[hit_mask]]

        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / max(1, top_k),
            dtype=torch.float32,
            device=device,
        )
        return topk_weights, topk_ids


class RoutingSimulator:
    """
    Token-to-Expert Routing Simulator.

    This class provides a framework for testing and comparing different
    routing strategies for MoE models. It can simulate routing behavior
    and collect statistics for analysis.
    """

    # Class-level registry of routing strategies
    _routing_strategies: dict[str, RoutingStrategy] = {
        # Basic routing strategies
        "uniform_random": DistributionBasedRouting(
            distribution="uniform", mean=0.0, std=1.0
        ),
        "normal_routing": DistributionBasedRouting(
            distribution="normal", mean=0.0, std=1.0
        ),
        "fixed_experts": FixedExpertsRouting(),
        "static_tail_hit": StaticTailHitRouting(),
    }

    @classmethod
    def register_strategy(cls, name: str, strategy: RoutingStrategy):
        """
        Register a custom routing strategy.

        Args:
            name: Name of the strategy
            strategy: RoutingStrategy instance
        """
        cls._routing_strategies[name] = strategy

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """
        Get list of available routing strategy names.

        Returns:
            List of available strategy names
        """
        return list(cls._routing_strategies.keys())

    @staticmethod
    def simulate_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        strategy_name: str,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate token-to-expert routing using the specified strategy.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            strategy_name: Name of the routing strategy to use
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        if strategy_name not in RoutingSimulator._routing_strategies:
            raise ValueError(
                f"Unknown routing strategy: {strategy_name}. "
                f"Available strategies: "
                f"{list(RoutingSimulator._routing_strategies.keys())}"
            )
        logger.warning_once(
            "Simulating MoE routing using a %s strategy. "
            "This should only be used for performance testing. "
            "Model outputs will not be valid.",
            strategy_name,
        )

        strategy = RoutingSimulator._routing_strategies[strategy_name]
        return strategy.route_tokens(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            indices_type=indices_type,
        )
