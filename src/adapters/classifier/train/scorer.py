import torch
from torch.nn.utils.rnn import pad_sequence

from ..preparation.trie import Encoded, Node

REPLY_START_NODE = -1
NodeLogProbs = dict[int, torch.Tensor]


def padded_sequences(encoded_batch: list[Encoded], device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [
        torch.tensor(encoded.prompt_ids + suffix, device=device)
        for encoded in encoded_batch
        for suffix in encoded.suffixes
    ]
    return pad_sequence(sequences, batch_first=True), pad_sequence(
        [torch.ones_like(sequence) for sequence in sequences], batch_first=True
    )


def example_hidden_rows(encoded: Encoded, option_hidden: torch.Tensor) -> dict[int, torch.Tensor]:
    prompt_length = len(encoded.prompt_ids)
    hidden_rows = {
        REPLY_START_NODE: option_hidden[0, encoded.reply_start_position],
        0: option_hidden[0, prompt_length - 1],
    }
    for node_index in encoded.row_nodes - {0}:
        option_index, step = encoded.trie.positions[node_index]
        hidden_rows[node_index] = option_hidden[option_index, prompt_length + step]
    return hidden_rows


def node_log_probs(model, encoded_batch: list[Encoded]) -> list[NodeLogProbs]:
    base_model = model.get_base_model()
    input_ids, attention_mask = padded_sequences(encoded_batch, base_model.device)
    hidden = base_model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    hidden_rows = [
        example_hidden_rows(encoded, option_hidden)
        for encoded, option_hidden in zip(
            encoded_batch, hidden.split([len(encoded.suffixes) for encoded in encoded_batch])
        )
    ]
    flat_hidden = torch.stack([vector for rows in hidden_rows for vector in rows.values()])
    log_probs = torch.log_softmax(base_model.lm_head(flat_hidden).float(), dim=-1).split(
        [len(rows) for rows in hidden_rows]
    )
    return [dict(zip(rows, example_log_probs)) for rows, example_log_probs in zip(hidden_rows, log_probs)]


def legal_tokens(encoded: Encoded, node_index: int, device) -> torch.Tensor:
    return torch.tensor(list(encoded.trie.nodes[node_index].children), device=device)


def branching_log_masses(current: NodeLogProbs, encoded: Encoded) -> dict[int, torch.Tensor]:
    return {
        node_index: torch.logsumexp(
            current[node_index][legal_tokens(encoded, node_index, current[node_index].device)], dim=-1
        )
        for node_index, node in enumerate(encoded.trie.nodes)
        if len(node.children) > 1
    }


def normalized_edge(node: Node, current: NodeLogProbs, log_masses: dict[int, torch.Tensor]):
    if node.parent not in log_masses:
        return 0.0
    return current[node.parent][node.token] - log_masses[node.parent]


def tree_log_q(current: NodeLogProbs, encoded: Encoded) -> torch.Tensor:
    log_masses = branching_log_masses(current, encoded)
    nodes = encoded.trie.nodes
    return torch.stack(
        [
            sum(normalized_edge(nodes[node_index], current, log_masses) for node_index in path)
            for path in encoded.trie.paths
        ]
    )
