from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MergePlan:
    table: str
    keys: tuple[str, ...]
    update_columns: tuple[str, ...]

    @property
    def predicate(self) -> str:
        return " AND ".join(f"target.{key} = source.{key}" for key in self.keys)


def build_merge_plan(table: str, keys: Sequence[str], columns: Iterable[str]) -> MergePlan:
    if not keys:
        raise ValueError("At least one merge key is required")
    key_tuple = tuple(keys)
    update_columns = tuple(column for column in columns if column not in key_tuple)
    return MergePlan(table=table, keys=key_tuple, update_columns=update_columns)

