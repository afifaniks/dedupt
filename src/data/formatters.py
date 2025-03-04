import re
from abc import ABC, abstractmethod
from typing import List


class StackFormatter(ABC):
    def __init__(self, from_top: bool, max_num_frames: int) -> None:
        self.from_top = from_top
        self.max_num_frames = max_num_frames

    @abstractmethod
    def format(self, stack: List[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class CppStackFormatter(StackFormatter):
    def __init__(self, from_top: bool, max_num_frames: int) -> None:
        super().__init__(from_top, max_num_frames)
        print("Using CppStackFormatter")

    def format(self, stack) -> str:
        # Remove duplicate frames
        stack = list(dict.fromkeys(stack))
        stack = [frame.lower() for frame in stack if frame.lower() != "none"]
        if self.max_num_frames > 0:
            if self.from_top:
                stack = stack[: self.max_num_frames]
            else:
                stack = stack[-self.max_num_frames :]

        stack = [
            re.sub(r"^_+gi_+", "", stack) for stack in stack
        ]  # Remove __GI__ if at the start
        stack = (
            "\n".join([f"{i+1}: {frame}" for i, frame in enumerate(stack)])
            .replace("_", " ")
            .lower()
        )
        stack = re.sub(r" +", " ", stack)

        return stack

    def name(self) -> str:
        return "cpp"


class CppStackFormatterPretrain(StackFormatter):
    def __init__(self, from_top: bool, max_num_frames: int) -> None:
        super().__init__(from_top, max_num_frames)
        print("Using CppStackFormatter Pretrain")

    def format(self, stack) -> str:
        # Convert a list of list of strings to a list of strings
        stack = [item for sublist in stack for item in sublist]
        # Remove duplicate frames
        stack = list(dict.fromkeys(stack))
        stack = [frame.lower() for frame in stack if frame.lower() != "none"]
        if self.max_num_frames > 0:
            if self.from_top:
                stack = stack[: self.max_num_frames]
            else:
                stack = stack[-self.max_num_frames :]

        stack = [
            re.sub(r"^_+gi_+", "", stack) for stack in stack
        ]  # Remove __GI__ if at the start
        stack = (
            "\n".join([f"{i+1}: {frame}" for i, frame in enumerate(stack)])
            .replace("_", " ")
            .lower()
        )
        stack = re.sub(r" +", " ", stack)

        return stack

    def name(self) -> str:
        return "cpp"


class JavaStackFormatter(StackFormatter):
    def __init__(self, from_top: bool, max_num_frames: int) -> None:
        super().__init__(from_top, max_num_frames)

    def format(self, stack) -> str:
        # Remove duplicate frames
        stack = list(dict.fromkeys(stack))
        stack = [frame for frame in stack if frame.lower() != "none"]
        if self.max_num_frames > 0:
            if self.from_top:
                stack = stack[: self.max_num_frames]
            else:
                stack = stack[-self.max_num_frames :]
        return (
            "\n".join(
                [
                    f"{i+1}: {frame}".replace(".", " ").strip()
                    for i, frame in enumerate(stack)
                ]
            )
            .lower()
            .strip()
        )

    def name(self) -> str:
        return "java"


class PretrainStackFormatter(StackFormatter):
    def __init__(self, from_top: bool, max_num_frames: int) -> None:
        super().__init__(from_top, max_num_frames)

    def _split_camel_mountain_case(self, text):
        # Split on transition from lowercase to uppercase or uppercase to lowercase
        words = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+", text)
        return words

    def format(self, stack) -> str:
        # Remove duplicate frames
        stack = list(dict.fromkeys(stack))

        # Seperate by camel case

        stack = [frame for frame in stack if frame.lower() != "none"]
        if self.max_num_frames > 0:
            if self.from_top:
                stack = stack[: self.max_num_frames]
            else:
                stack = stack[-self.max_num_frames :]

        if stack[-1].startswith("EXC"):
            exception = stack[-1].replace("EXC", "")
            stack = [frame for frame in stack[:-1]]
            stack.append(exception)
        else:
            stack = [frame for frame in stack]

        stack = [self._split_camel_mountain_case(frame) for frame in stack]

        return "\n".join(
            [f"{i+1}: {' '.join(frame).lower()}" for i, frame in enumerate(stack)]
        ).strip()

    def name(self) -> str:
        return "pretrain"


# Make a factory of fomatters
def get_formatter(name: str, num_frames: int) -> StackFormatter:
    print("Selected formatter:", name, "num_frames:", num_frames)
    if name == "cpp":
        return CppStackFormatter(False, num_frames)
    elif name == "cpp_pretrain":
        return CppStackFormatterPretrain(False, num_frames)
    elif name == "java":
        return JavaStackFormatter(False, num_frames)
    elif name == "java_pretrain":
        return PretrainStackFormatter(False, num_frames)
    else:
        raise ValueError(f"Unknown formatter name: {name}")
