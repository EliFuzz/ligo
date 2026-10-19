import random
from dataclasses import dataclass


@dataclass
class Example:
    identifier: str
    state: str
    instruction: str
    options: list[str]
    target: list[float]
    kind: str

    @property
    def gold(self) -> int:
        return max(range(len(self.target)), key=self.target.__getitem__)

    def shuffled(self) -> "Example":
        if self.kind == "score":
            return self
        order = list(range(len(self.options)))
        random.Random(self.identifier).shuffle(order)
        return Example(
            self.identifier,
            self.state,
            self.instruction,
            [self.options[index] for index in order],
            [self.target[index] for index in order],
            self.kind,
        )
