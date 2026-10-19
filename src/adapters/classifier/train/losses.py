from dataclasses import dataclass

import torch

from ..preparation import Example
from ..preparation.trie import Encoded
from .scorer import REPLY_START_NODE, NodeLogProbs, legal_tokens, tree_log_q

TOP_K = 64


@dataclass
class DistributionSummary:
    top_indices: torch.Tensor
    top_log_probs: torch.Tensor
    tail_log_mass: torch.Tensor


@dataclass
class AnchorReference:
    legal_log_mass: torch.Tensor
    illegal_log_mass: torch.Tensor
    illegal: DistributionSummary


@dataclass
class ReferenceTargets:
    reply_start: DistributionSummary
    gold_leaf: DistributionSummary
    anchors: list[AnchorReference]


def log_mass_outside(log_probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(log_probs.index_fill(0, indices, float("-inf")), dim=-1)


def summarize(log_probs: torch.Tensor) -> DistributionSummary:
    top = torch.topk(log_probs, TOP_K)
    return DistributionSummary(top.indices, top.values, log_mass_outside(log_probs, top.indices))


def top_k_tail_kl(reference: DistributionSummary, log_probs: torch.Tensor) -> torch.Tensor:
    top_divergence = (
        reference.top_log_probs.exp() * (reference.top_log_probs - log_probs[reference.top_indices])
    ).sum()
    tail_log_mass = log_mass_outside(log_probs, reference.top_indices)
    tail_divergence = reference.tail_log_mass.exp() * (reference.tail_log_mass - tail_log_mass)
    return (top_divergence + tail_divergence).clamp(min=0.0)


def split_mass(log_probs: torch.Tensor, legal: torch.Tensor):
    illegal_log_probs = log_probs.index_fill(0, legal, float("-inf"))
    illegal_mass = torch.logsumexp(illegal_log_probs, dim=-1)
    return torch.logsumexp(log_probs[legal], dim=-1), illegal_mass, illegal_log_probs - illegal_mass


def anchor_reference(log_probs: torch.Tensor, legal: torch.Tensor) -> AnchorReference:
    legal_mass, illegal_mass, illegal_log_probs = split_mass(log_probs, legal)
    return AnchorReference(legal_mass, illegal_mass, summarize(illegal_log_probs))


def anchor_node_loss(log_probs: torch.Tensor, reference: AnchorReference, legal: torch.Tensor) -> torch.Tensor:
    legal_mass, illegal_mass, illegal_log_probs = split_mass(log_probs, legal)
    mass_kl = reference.legal_log_mass.exp() * (reference.legal_log_mass - legal_mass) + (
        reference.illegal_log_mass.exp() * (reference.illegal_log_mass - illegal_mass)
    )
    return mass_kl.clamp(min=0.0) + top_k_tail_kl(reference.illegal, illegal_log_probs)


def reference_targets(reference_log_probs: NodeLogProbs, encoded: Encoded) -> ReferenceTargets:
    return ReferenceTargets(
        reply_start=summarize(reference_log_probs[REPLY_START_NODE]),
        gold_leaf=summarize(reference_log_probs[encoded.gold_leaf]),
        anchors=[
            anchor_reference(
                reference_log_probs[node_index],
                legal_tokens(encoded, node_index, reference_log_probs[node_index].device),
            )
            for node_index in encoded.anchor_nodes
        ],
    )


def ranked_probability_score(log_q: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    cumulative_difference = log_q.exp().cumsum(0) - target_distribution.cumsum(0)
    return (cumulative_difference[:-1] ** 2).sum() / (len(log_q) - 1)


def classifier_loss(
    current: NodeLogProbs,
    reference: ReferenceTargets,
    encoded: Encoded,
    example: Example,
    regularization_weight: float,
    ordinal_weight: float,
) -> torch.Tensor:
    log_q = tree_log_q(current, encoded)
    target_distribution = torch.tensor(example.target, device=log_q.device)
    tree_loss = -(target_distribution * log_q).sum()
    ordinal_loss = ranked_probability_score(log_q, target_distribution) if example.kind == "score" else 0.0
    anchor_loss = sum(
        anchor_node_loss(current[node_index], anchor, legal_tokens(encoded, node_index, log_q.device))
        for node_index, anchor in zip(encoded.anchor_nodes, reference.anchors)
    )
    position_loss = top_k_tail_kl(reference.reply_start, current[REPLY_START_NODE]) + top_k_tail_kl(
        reference.gold_leaf, current[encoded.gold_leaf]
    )
    return (
        tree_loss
        + ordinal_weight * ordinal_loss
        + regularization_weight * (anchor_loss / len(reference.anchors) + position_loss / 2)
    )
