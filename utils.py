from datetime import datetime
from dateutil import parser
import pandas as pd

def normalize_datetime(date_input, output_format="%d-%b-%Y"):
    """
    Convert various date/time string formats into a standard date string.
    Returns the formatted string or None if parsing fails.
    """
    if pd.isna(date_input) or str(date_input).strip() == "":
        return None

    if isinstance(date_input, datetime):
        return date_input.strftime(output_format)

    try:
        dt = parser.parse(str(date_input), dayfirst=True, fuzzy=True)
        return dt.strftime(output_format)
    except Exception:
        return None
