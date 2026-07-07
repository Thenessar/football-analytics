"""Shared append + active-flag Delta write strategy (§15.4 Option C).

Used by the prediction writer (`football_analytics/inference.py`) and the
simulation writer (`football_analytics/simulation.py`): rows with the same
deterministic set id replace themselves, appends preserve audit history, and
older active sets for the same flip key are switched off so exactly one set
stays active.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import pandas as pd


def _sql_literal(value: Any) -> str:
    """Renders a flip-key value for a WHERE clause: ints bare, rest quoted."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) or (hasattr(value, "item") and isinstance(value.item(), int)):
        return str(int(value))
    return f"'{value}'"


def write_active_flag_records(
    spark: Any,
    records: pd.DataFrame,
    *,
    table: str,
    ensure_table: Callable[[Any, str], None],
    column_types: Mapping[str, str],
    set_id_column: str,
    flip_key_columns: Sequence[str],
    active_flag_column: str,
) -> dict[str, int]:
    """Append + active-flag write; returns written/deactivated counts.

    1. Rows sharing a set id with incoming records are deleted first, so
       identical reruns replace themselves instead of duplicating.
    2. The append casts to the DDL types (Spark infers LONG for Python ints)
       and uses mergeSchema so additive column changes reach older tables.
    3. Older sets for the same flip key are switched inactive, leaving
       exactly one active set per key.
    """

    if records.empty:
        return {"written": 0, "deactivated_sets": 0}

    ensure_table(spark, table)
    frame = spark.createDataFrame(records)
    for column, data_type in column_types.items():
        if column in frame.columns:
            frame = frame.withColumn(column, frame[column].cast(data_type))

    set_ids = sorted(records[set_id_column].unique())
    quoted = ", ".join(f"'{set_id}'" for set_id in set_ids)
    spark.sql(f"DELETE FROM {table} WHERE {set_id_column} IN ({quoted})")
    frame.write.format("delta").mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(table)

    deactivated = 0
    keys = records[[*flip_key_columns, set_id_column]].drop_duplicates()
    for row in keys.itertuples(index=False):
        conditions = "\n              AND ".join(
            f"{column} = {_sql_literal(getattr(row, column))}"
            for column in flip_key_columns
        )
        spark.sql(
            f"""
            UPDATE {table}
            SET {active_flag_column} = false
            WHERE {conditions}
              AND {set_id_column} != '{getattr(row, set_id_column)}'
              AND {active_flag_column} = true
            """
        )
        deactivated += 1
    return {"written": len(records), "deactivated_sets": deactivated}
