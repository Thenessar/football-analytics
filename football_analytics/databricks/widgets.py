def get_widget_or_default(dbutils, name: str, default: str) -> str:
    try:
        value = dbutils.widgets.get(name)
    except Exception:
        return default
    return value or default

