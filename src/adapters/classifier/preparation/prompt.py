from dataclasses import dataclass

from .example import Example


@dataclass
class RenderedPrompt:
    text: str
    reply_start: str


def render(tokenizer, example: Example) -> RenderedPrompt:
    option_lines = [f"[{number}] {option.strip()}" for number, option in enumerate(example.options, start=1)]
    user_message = "\n".join(
        [
            example.state.strip(),
            "",
            example.instruction.strip(),
            "",
            "Options:",
            *option_lines,
            "",
            "Answer only with its bracketed numeric identifier.",
        ]
    )
    reply_start = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return RenderedPrompt(text=reply_start + "Best answer: [", reply_start=reply_start)
