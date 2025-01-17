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
        stack = [frame for frame in stack if frame.lower() != "none"]
        if self.max_num_frames > 0:
            if self.from_top:
                stack = stack[: self.max_num_frames]
            else:
                stack = stack[-self.max_num_frames :]

        stack = "\n".join([f"{i+1}: {frame}" for i, frame in enumerate(stack)]).replace(
            "_", " "
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
        return "\n".join([f"{i+1}: {frame}" for i, frame in enumerate(stack)])

    def name(self) -> str:
        return "java"


# Make a factory of fomatters
def get_formatter(name: str) -> StackFormatter:
    if name == "cpp":
        return CppStackFormatter(False, 100)
    elif name == "java":
        return JavaStackFormatter(False, 10)
    else:
        raise ValueError(f"Unknown formatter name: {name}")
