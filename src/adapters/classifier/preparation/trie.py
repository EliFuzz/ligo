from dataclasses import dataclass, field
from os.path import commonprefix

from .example import Example
from .prompt import render


@dataclass
class Node:
    parent: int
    token: int
    children: dict[int, int] = field(default_factory=dict)


@dataclass
class Trie:
    nodes: list[Node]
    paths: list[list[int]]
    positions: dict[int, tuple[int, int]]


@dataclass
class Encoded:
    prompt_ids: list[int]
    reply_start_position: int
    suffixes: list[list[int]]
    trie: Trie
    gold_option: int

    @property
    def anchor_nodes(self) -> list[int]:
        return [self.trie.nodes[node_index].parent for node_index in self.trie.paths[self.gold_option]]

    @property
    def gold_leaf(self) -> int:
        return self.trie.paths[self.gold_option][-1]

    @property
    def row_nodes(self) -> set[int]:
        branching_nodes = {node_index for node_index, node in enumerate(self.trie.nodes) if len(node.children) > 1}
        return branching_nodes | set(self.anchor_nodes) | {self.gold_leaf}


def build_trie(suffixes: list[list[int]]) -> Trie:
    nodes = [Node(parent=-1, token=-1)]
    paths = []
    positions = {}
    for option_index, suffix in enumerate(suffixes):
        current = 0
        path = []
        for step, token in enumerate(suffix):
            if token not in nodes[current].children:
                nodes[current].children[token] = len(nodes)
                nodes.append(Node(parent=current, token=token))
            current = nodes[current].children[token]
            positions.setdefault(current, (option_index, step))
            path.append(current)
        paths.append(path)
    return Trie(nodes=nodes, paths=paths, positions=positions)


def encode(tokenizer, example: Example) -> Encoded:
    def token_ids(text: str) -> list[int]:
        return tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids

    rendered = render(tokenizer, example)
    prompt_ids = token_ids(rendered.text)
    full_sequences = [token_ids(f"{rendered.text}{number}]") for number in range(1, len(example.options) + 1)]
    shared_length = min(len(commonprefix([prompt_ids, sequence])) for sequence in full_sequences)
    suffixes = [sequence[shared_length:] for sequence in full_sequences]
    reply_start_length = len(commonprefix([token_ids(rendered.reply_start), prompt_ids]))
    maximum_length = tokenizer.model_max_length - max(map(len, suffixes))
    removed_tokens = max(0, shared_length - maximum_length)
    return Encoded(
        prompt_ids=prompt_ids[max(0, shared_length - maximum_length) : shared_length],
        reply_start_position=max(0, min(reply_start_length - removed_tokens, maximum_length) - 1),
        suffixes=suffixes,
        trie=build_trie(suffixes),
        gold_option=example.gold,
    )
