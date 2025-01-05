import json
import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from data.objects import Stack
from data.readers import read_stack


class StackLoader(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, id: int) -> Stack:
        raise NotImplementedError


class DirectoryStackLoader(StackLoader):
    def __init__(self, *dirs: str, frames_field: str = "frames"):
        self.dirs = list(dirs)
        self.id_dir = {}
        self.frames_field = frames_field

    def init(self, *dirs: str):
        self.dirs += list(dirs)

    def add(self, directory: str):
        self.dirs.append(directory)

    def name(self) -> str:
        return ("rec" if self.frames_field == "frames" else "notrec") + "_loader"

    @lru_cache(maxsize=300_000)
    def __call__(self, id: int) -> Stack:
        if id not in self.id_dir:
            for d in self.dirs:
                if os.path.exists(f"{d}/{id}.json"):
                    self.id_dir[id] = d
                    break
        if id in self.id_dir:
            return read_stack(f"{self.id_dir[id]}/{id}.json", self.frames_field)
        return None


class JsonStackLoader(StackLoader):
    def __init__(self, reports_path: str):
        self.reports_path = reports_path
        self.reports = {}

        raw_reports = json.load(open(reports_path, "r"))
        for report in raw_reports:
            if report is None:
                continue

            stacktrace = report["stacktrace"]
            if isinstance(stacktrace, list):
                stacktrace = stacktrace[0]
            st_id = report["bug_id"]
            exception = stacktrace["exception"] or []
            if isinstance(exception, str):
                exception = [exception]

            raw_frames = stacktrace["frames"]
            frames = [frame["function"] for frame in raw_frames]

            self.reports[st_id] = Stack(st_id, report["creation_ts"], exception, frames)

    def name(self) -> str:
        return "json_loader"

    def __call__(self, id: int) -> Stack:
        return self.reports[id]


class JsonStackLoaderForCpp(StackLoader):
    def __init__(self, reports_path: str):
        self.reports_path = reports_path
        self.reports = {}

        raw_reports = json.load(open(reports_path, "r"))
        for report in raw_reports:
            if report is None:
                continue

            stacktrace = report["stacktrace"]
            if isinstance(stacktrace, list):
                stacktrace = stacktrace[0]
            st_id = report["bug_id"]
            exception = stacktrace["exception"] or []
            if isinstance(exception, str):
                exception = [exception]

            raw_frames = stacktrace["frames"]
            frames = []

            for frame in raw_frames:
                if frame.get("function", None):
                    function_name = frame["function"]
                    # Normalize the function name
                    function_name = function_name.lower()
                    function_name = re.sub(
                        r"^__gi__", "", function_name
                    )  # Remove __GI__ if at the start
                    function_name = re.sub(
                        r"^_+", "", function_name
                    )  # Remove leading underscores
                    function_name = re.sub(
                        r"_+", "_", function_name
                    )  # Replace double underscores

                    # Process the file path, if available
                    file_path = frame.get("file", None)
                    if file_path:
                        # Normalize the file path (e.g., remove special characters and simplify)
                        file_path = re.sub(
                            r"[^a-zA-Z0-9/\.]", "", file_path
                        )  # Remove non-alphanumeric except '/'
                        file_path = file_path.lower()  # Convert to lowercase
                        file_path = file_path.split("/")[-1]

                    # # Process the dylib path, if available
                    dylib_path = frame.get("dylib", None)
                    if dylib_path:
                        # Normalize the dylib path
                        dylib_path = re.sub(
                            r"[^a-zA-Z0-9/\.]", "", dylib_path
                        )  # Remove non-alphanumeric except '/'
                        dylib_path = dylib_path.lower()  # Convert to lowercase
                        dylib_path = dylib_path.split("/")[-1]

                    # Construct the normalized representation
                    normalized_frame = function_name
                    if file_path:
                        normalized_frame += f" at {file_path}"
                    elif dylib_path:
                        normalized_frame += f" at {dylib_path}"

                    frames.append(normalized_frame)

            # frames = [frame["function"] for frame in raw_frames]

            self.reports[st_id] = Stack(st_id, report["creation_ts"], exception, frames)

    def name(self) -> str:
        return "json_loader"

    def __call__(self, id: int) -> Stack:
        return self.reports[id]