import os
import sys
import argparse
from pathlib import Path

# Parse .env manually to avoid dependency issues if run outside of venv
def load_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    
    parser = argparse.ArgumentParser(description="Query Databricks SQL Warehouse")
    parser.add_argument("--query", required=True, help="SQL query to execute")
    parser.add_argument("--limit", type=int, default=50, help="Limit output rows")
    args = parser.parse_args()
    
    host = os.environ.get("DATABRICKS_HOST")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH")
    token = os.environ.get("DATABRICKS_TOKEN")
    
    if not host or not http_path or not token:
        print("Error: Missing required environment variables. Ensure .env is populated with host, http_path, and token.")
        sys.exit(1)
        
    try:
        from databricks import sql
        import pandas as pd
    except ImportError:
        print("Error: databricks-sql-connector and pandas are required. Run in the virtual environment (.venv).")
        sys.exit(1)
        
    print(f"Connecting to Databricks SQL Warehouse...")
    try:
        with sql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=token
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(args.query)
                try:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    if not rows:
                        print("Query returned no rows.")
                        return
                    df = pd.DataFrame.from_records(rows, columns=columns)
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', 1000)
                    print(df.head(args.limit))
                    print(f"\nTotal rows returned: {len(df)}")
                except Exception as exc:
                    print("Query executed, but could not retrieve description/rows (maybe a DDL/DML statement).")
                    print(exc)
    except Exception as exc:
        print(f"Connection/Execution failed: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
