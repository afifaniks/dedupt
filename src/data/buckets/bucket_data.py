import json
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional

from data.stack_loader import (
    JsonStackLoader,
    JsonStackLoaderForCpp,
    JsonStackLoaderForCppMulti,
    JsonStackLoaderJavaMulti,
    StackLoader,
)

StackAdditionEvent = namedtuple("StackAdditionEvent", "id st_id is_id ts label")


class BucketData(ABC):
    def __init__(
        self,
        name: str,
        train_days: int,
        test_days: int,
        warmup_days: int,
        val_days: int,
        reports_path: str = None,
        sep: str = ".",
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.warmup_days = warmup_days
        self.val_days = val_days

        self.full_name = name
        self.name = name.split("_")[0]
        self.reports_path = reports_path
        self.sep = sep

    @abstractmethod
    def load(self) -> "BucketData":
        raise NotImplementedError

    @abstractmethod
    def events(self) -> List[StackAdditionEvent]:
        raise NotImplementedError

    @abstractmethod
    def stack_loader(self, multi_stack: Optional[bool]) -> StackLoader:
        raise NotImplementedError

    @abstractmethod
    def get_dup_id(self, st_id: int) -> int:
        raise NotImplementedError


class OtherBucketData(BucketData):
    def __init__(
        self,
        name: str,
        reports_path: str,
        train_days: int,
        test_days: int,
        warmup_days: int,
        val_days: int,
        sep: str = ".",
        lang: str = "",
        multi_stack: bool = False,
    ):
        super().__init__(
            name, train_days, test_days, warmup_days, val_days, reports_path, sep
        )
        self.actions = None
        self.lang = lang
        self.multi_stack = multi_stack
        self.dup_id_by_bug_id = {}

    def load(self) -> "OtherBucketData":
        self.actions = []
        raw_reports = json.load(open(self.reports_path, "r"))

        day_secs = 60 * 60 * 24
        first_ts = min([report["creation_ts"] for report in raw_reports])

        for report in raw_reports:
            if report is None:
                continue

            st_id = report["bug_id"]
            # if report["dup_id"] is not None:
            #     print("Found dup_id", report["dup_id"])
            dup_id = report["dup_id"] or st_id
            ts = (report["creation_ts"] - first_ts) / day_secs
            self.dup_id_by_bug_id[st_id] = dup_id

            addition_event = StackAdditionEvent(
                len(self.actions), st_id, dup_id, ts, True
            )
            self.actions.append(addition_event)

        return self

    def events(self) -> List[StackAdditionEvent]:
        return self.actions

    def stack_loader(self, multi_stack: bool = False) -> StackLoader:
        if multi_stack and self.lang == "java":
            return JsonStackLoaderJavaMulti(self.reports_path)
        if self.lang == "cpp":
            return JsonStackLoaderForCpp(self.reports_path)
        if self.lang == "cpp_pretrain":
            return JsonStackLoaderForCppMulti(self.reports_path)

        return JsonStackLoader(self.reports_path)

    def get_dup_id(self, st_id: int) -> int:
        return self.dup_id_by_bug_id[st_id]
