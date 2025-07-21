import pandas as pd
import numpy as np
import re
from datetime import datetime
import uuid
import calendar


def convert_duration_to_minutes(x):

    str_match = re.search(r"(?:(\d+)h)?\s*(?:(\d+)m)?", x)
    h = int(str_match[1] or 0)
    m = int(str_match[2] or 0)
    return h * 60 + m


def convert_timestamps(x):

    return datetime.strptime(x, "%H:%M").time()


def get_connection_flights(x):
    strs = x[0]
    if strs.isdigit():
        return int(strs)
    else:
        return 0


def process_raw_data(df):

    df = df.copy()

    df["date_of_booking"] = pd.to_datetime(df["Date of Booking"], format="%d/%m/%Y")
    df["date_of_journey"] = pd.to_datetime(df["Date of Journey"], format="%d/%m/%Y")
    df[["airline", "flight_code", "class"]] = df["Airline-Class"].str.split(
        "\n", expand=True
    )
    df["connections"] = df["Total Stops"].apply(get_connection_flights)
    df["flight_code"] = df["flight_code"].apply(
        lambda x: "-".join(part.strip() for part in str(x).split("-")[:2])
    )

    df["duration_minutes"] = df["Duration"].apply(convert_duration_to_minutes)
    df[["departure_time", "source_city"]] = df["Departure Time"].str.split(
        "\n", expand=True
    )
    df[["arrival_time", "destination_city"]] = df["Arrival Time"].str.split(
        "\n", expand=True
    )

    df["departure_time"] = df["departure_time"].apply(convert_timestamps)
    df["arrival_time"] = df["arrival_time"].apply(convert_timestamps)

    df["price"] = df["Price"].str.replace(",", "").astype(float)
    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    df = df.drop(
        [
            "Date of Booking",
            "Date of Journey",
            "Airline-Class",
            "Departure Time",
            "Arrival Time",
            "Duration",
            "Total Stops",
            "Price",
        ],
        axis=1,
    )
    return df


def day_of_month_category(date):
    ratio = date.day / date.days_in_month
    if ratio <= 1 / 3:
        return "early"
    elif ratio <= 2 / 3:
        return "middle"
    else:
        return "end"


def minute_category(time):
    minute = time.minute
    if minute < 15:
        return "00"
    elif minute < 30:
        return "15"
    elif minute < 45:
        return "30"
    else:
        return "45"


def time_of_day_category(time):
    hour = time.hour
    if 0 <= hour < 5:
        return "Late_Night"
    elif 5 <= hour < 8:
        return "Early_Morning"
    elif 8 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def convert_data_to_ml_format(df):

    df = df.drop("flight_code", axis=1)
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    date_cols = [col for col in df.columns if col and "date" in col]
    time_cols = [col for col in df.columns if col and "time" in col]

    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols
        and "date" not in col
        and "id" not in col
        and "time" not in col
        and "code" not in col
    ]

    dummy_dfs = []

    for col in categorical_cols:
        dummy = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
        dummy_dfs.append(dummy)

    df = pd.concat([df] + dummy_dfs, axis=1)

    df = df.drop(columns=categorical_cols)

    df.columns = [col.strip().replace(" ", "_") for col in df.columns]

    date_cat_cols = []

    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        df[f"{col}_year"] = df[col].dt.year.astype(float)
        df[f"{col}_day_of_week"] = df[col].dt.dayofweek
        df[f"{col}_day_of_week_cat"] = df[f"{col}_day_of_week"].apply(
            lambda x: calendar.day_name[x]
        )
        df[f"{col}_is_weekend"] = df[f"{col}_day_of_week"].isin([5, 6]).astype(int)
        df[f"{col}_day_of_month_category"] = df[col].apply(day_of_month_category)

        date_cat_cols.extend([f"{col}_day_of_month_category", f"{col}_day_of_week_cat"])

        df = df.drop([col, f"{col}_day_of_week"], axis=1)

    dummy_cols_dt = pd.get_dummies(
        df[date_cat_cols], prefix=date_cat_cols, drop_first=True
    ).astype(int)
    df = pd.concat([df.drop(columns=date_cat_cols), dummy_cols_dt], axis=1)

    df.columns = [col.strip() for col in df.columns]

    time_cat_cols = []

    for col in time_cols:
        df[col] = pd.to_datetime(df[col], format="%H:%M:%S").dt.time

        df[f"{col}_minute_category"] = df[col].apply(minute_category)
        df[f"{col}_time_of_day_category"] = df[col].apply(time_of_day_category)

        time_cat_cols.extend([f"{col}_time_of_day_category", f"{col}_minute_category"])

        df = df.drop(col, axis=1)

    dummy_cols_time = pd.get_dummies(
        df[time_cat_cols], prefix=time_cat_cols, drop_first=True
    ).astype(int)
    df = pd.concat([df.drop(columns=time_cat_cols), dummy_cols_time], axis=1)

    df.columns = [col.strip() for col in df.columns]

    return df
