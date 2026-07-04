# Databricks notebook source
import json
import os
from pathlib import PurePosixPath
import subprocess
import sys
import tempfile


dbutils.widgets.text("workspace_root_path", "")
dbutils.widgets.text("workspace_file_path", "")
dbutils.widgets.text("sql_warehouse_id", "")
dbutils.widgets.text("catalog", "football_analytics")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")


def widget_value(name: str, default: str = "") -> str:
    value = dbutils.widgets.get(name).strip()
    return value or default


def option_value(option) -> str:
    try:
        if option.isDefined():
            return str(option.get())
    except Exception:
        pass
    try:
        return str(option.get())
    except Exception:
        return ""


def context_option(context, name: str) -> str:
    try:
        return option_value(getattr(context, name)())
    except Exception:
        return ""


def workspace_fs_path(path: str) -> str:
    if path.startswith("/Workspace/"):
        return path
    if path.startswith("/"):
        return f"/Workspace{path}"
    return path


def candidate_roots_from_workspace_path(path: str) -> list[str]:
    workspace_path = PurePosixPath(workspace_fs_path(path).rstrip("/"))
    candidates = [str(workspace_path)]
    if workspace_path.name == "files":
        candidates.append(str(workspace_path.parent))
    elif workspace_path.suffix:
        candidates.append(str(workspace_path.parent))
    return candidates


def notebook_root_candidate(context) -> str:
    notebook_path = option_value(context.notebookPath())
    if not notebook_path:
        return ""
    notebook_fs_path = workspace_fs_path(notebook_path)
    return str(PurePosixPath(notebook_fs_path).parent.parent)


def find_dbt_project_dir(context) -> str:
    workspace_file_path = widget_value("workspace_file_path")
    candidates = []
    if workspace_file_path:
        candidates.extend(
            f"{candidate_root}/dbt"
            for candidate_root in candidate_roots_from_workspace_path(workspace_file_path)
        )
    root_candidate = notebook_root_candidate(context)
    if root_candidate:
        candidates.append(f"{root_candidate}/dbt")

    candidates = list(dict.fromkeys(candidates))
    for candidate in candidates:
        if os.path.exists(f"{candidate}/dbt_project.yml"):
            return candidate

    raise FileNotFoundError(
        "Could not locate dbt_project.yml. Checked: " + ", ".join(candidates)
    )


def workspace_root(context) -> str:
    root_path = widget_value("workspace_root_path")
    if root_path:
        return workspace_fs_path(root_path).rstrip("/")

    workspace_file_path = widget_value("workspace_file_path")
    if workspace_file_path:
        for candidate_root in candidate_roots_from_workspace_path(workspace_file_path):
            if PurePosixPath(candidate_root).name != "files":
                return candidate_root

    root_candidate = notebook_root_candidate(context)
    if root_candidate:
        return root_candidate.rstrip("/")

    raise ValueError("Missing workspace_root_path and unable to infer bundle workspace root")


def required_value(name: str, value: str) -> str:
    if not value:
        raise ValueError(f"Missing required Databricks context value: {name}")
    return value


def yaml_string(value: str) -> str:
    return json.dumps(value)


def run_dbt_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )


context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

catalog = widget_value("catalog", "football_analytics")
bronze_schema = widget_value("bronze_schema", "bronze")
silver_schema = widget_value("silver_schema", "silver")
gold_schema = widget_value("gold_schema", "gold")
bundle_workspace_root = workspace_root(context)
sql_warehouse_id = widget_value("sql_warehouse_id")
host = os.environ.get("DATABRICKS_HOST", "").strip() or context_option(context, "apiUrl")
token = os.environ.get("DATABRICKS_TOKEN", "").strip() or context_option(context, "apiToken")
http_path = os.environ.get("DATABRICKS_HTTP_PATH", "").strip()
if not http_path and sql_warehouse_id:
    http_path = f"/sql/1.0/warehouses/{sql_warehouse_id}"

if not http_path:
    sql_warehouse_id = required_value("sql_warehouse_id", sql_warehouse_id)
host = required_value("DATABRICKS_HOST", host)
token = required_value("DATABRICKS_TOKEN", token)
http_path = required_value("DATABRICKS_HTTP_PATH", http_path)
dbt_project_dir = find_dbt_project_dir(context)

os.environ.update(
    {
        "DBT_FOOTBALL_CATALOG": catalog,
        "DBT_FOOTBALL_BRONZE_SCHEMA": bronze_schema,
        "DBT_FOOTBALL_SILVER_SCHEMA": silver_schema,
        "DBT_FOOTBALL_GOLD_SCHEMA": gold_schema,
        "DBT_WORKSPACE_ROOT_PATH": bundle_workspace_root,
        "DATABRICKS_HOST": host,
        "DATABRICKS_HTTP_PATH": http_path,
        "DATABRICKS_TOKEN": token,
    }
)

dbt_vars = json.dumps(
    {
        "catalog": catalog,
        "bronze_schema": bronze_schema,
        "silver_schema": silver_schema,
        "gold_schema": gold_schema,
    }
)

with tempfile.TemporaryDirectory(prefix="dbt-profiles-") as profiles_dir:
    profile_path = f"{profiles_dir}/profiles.yml"
    with open(profile_path, "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "football_analytics_databricks:",
                    "  target: bundle_cluster",
                    "  outputs:",
                    "    bundle_cluster:",
                    "      type: databricks",
                    f"      catalog: {yaml_string(catalog)}",
                    f"      schema: {yaml_string(silver_schema)}",
                    f"      host: {yaml_string(host)}",
                    f"      http_path: {yaml_string(http_path)}",
                    "      token: \"{{ env_var('DATABRICKS_TOKEN') }}\"",
                    "      threads: 4",
                    "",
                ]
            )
        )

    command = [
        sys.executable,
        "-c",
        "from dbt.cli.main import cli; cli()",
        "build",
        "--project-dir",
        dbt_project_dir,
        "--profiles-dir",
        profiles_dir,
        "--select",
        "tag:python",
        "--vars",
        dbt_vars,
        "--no-use-colors",
    ]
    print("Running dbt Python models with selector: tag:python")
    completed = run_dbt_command(command)
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"dbt Python model build failed with exit code {completed.returncode}")
    print("Success! dbt Python models built successfully.")
