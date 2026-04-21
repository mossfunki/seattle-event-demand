"""
model.py
Trains an XGBoost model to predict hourly traffic volume lift around Seattle event venues.
Uses event features (attendance bucket, venue proximity, day-of-week, hour) + baseline traffic.
Outputs: outputs/volume_lift_predictions.csv, outputs/model_metrics.json, outputs/feature_importance.csv
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

VENUE_COORDS = {
    "Lumen Field":          (47.5952, -122.3316),
    "T-Mobile Park":        (47.5914, -122.3324),
    "Climate Pledge Arena": (47.6220, -122.3542),
    "Convention Center":    (47.6101, -122.3326),
    "Lake Washington":      (47.5950, -122.2830),
}

STATION_COORDS = {
    110: (47.578,  -122.327),   # I-90 at 4th Ave S
    102: (47.624,  -122.330),   # I-5 NB Mercer
    108: (47.572,  -122.326),   # I-5 SB Spokane
    215: (47.608,  -122.342),   # SR-99 Western Ave
    320: (47.498,  -122.194),   # I-405 Renton
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Approximate distance in km between two lat/lon points."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_data():
    traffic = pd.read_csv(DATA_DIR / "wsdot_counts.csv", parse_dates=["datetime"])
    events  = pd.read_csv(DATA_DIR / "events.csv",       parse_dates=["date"])
    return traffic, events


def compute_baseline(traffic: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline volume per station per hour-of-day-of-week."""
    traffic["hour"] = traffic["datetime"].dt.hour
    traffic["dow"]  = traffic["datetime"].dt.dayofweek
    baseline = (traffic.groupby(["station_id", "dow", "hour"])["volume"]
                .median()
                .reset_index()
                .rename(columns={"volume": "baseline_volume"}))
    return baseline


def build_feature_matrix(traffic: pd.DataFrame, events: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    traffic = traffic.merge(baseline, on=["station_id", "dow", "hour"], how="left")
    traffic["volume_lift"] = (traffic["volume"] - traffic["baseline_volume"]) / (traffic["baseline_volume"] + 1)

    rows = []
    for _, row in traffic.iterrows():
        dt         = row["datetime"]
        station_id = row["station_id"]
        s_lat, s_lon = STATION_COORDS.get(station_id, (47.6, -122.33))

        # Find events within ±12 hours of this timestamp
        nearby_events = events[
            (events["date"] >= dt - pd.Timedelta("12h")) &
            (events["date"] <= dt + pd.Timedelta("12h"))
        ]

        if len(nearby_events):
            ev = nearby_events.iloc[0]
            v_lat, v_lon = VENUE_COORDS.get(ev["venue"], (47.60, -122.33))
            dist_km      = haversine_km(s_lat, s_lon, v_lat, v_lon)
            hours_to_ev  = (ev["date"] - dt).total_seconds() / 3600
            attendance   = int(ev["attendance"])
            att_large    = int(attendance >= 40_000)
            att_medium   = int(15_000 <= attendance < 40_000)
            att_small    = int(0 < attendance < 15_000)
        else:
            dist_km = 99; hours_to_ev = 99
            attendance = att_large = att_medium = att_small = 0

        rows.append({
            "datetime":        dt,
            "station_id":      station_id,
            "volume":          row["volume"],
            "baseline_volume": row["baseline_volume"],
            "volume_lift":     row["volume_lift"],
            "hour":            row["hour"],
            "dow":             row["dow"],
            "is_weekend":      int(row["dow"] >= 5),
            "attendance":      attendance,
            "att_large":       att_large,
            "att_medium":      att_medium,
            "att_small":       att_small,
            "dist_km":         dist_km,
            "hours_to_event":  hours_to_ev,
            "proximity_score": max(0, 1 - dist_km / 10) * (1 if abs(hours_to_ev) < 4 else 0.3),
        })

    return pd.DataFrame(rows)


FEATURES = ["hour", "dow", "is_weekend", "att_large", "att_medium", "att_small",
            "dist_km", "hours_to_event", "proximity_score", "baseline_volume"]


def train_and_evaluate(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + ["volume_lift"])
    df = df.sort_values("datetime").reset_index(drop=True)

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.04,
        subsample=0.85, random_state=42
    )
    model.fit(train[FEATURES], train["volume_lift"])

    test = test.copy()
    test["predicted_lift"] = model.predict(test[FEATURES])
    test["predicted_volume"] = test["baseline_volume"] * (1 + test["predicted_lift"])

    mape = mean_absolute_percentage_error(
        test["volume"].clip(1), test["predicted_volume"].clip(1)
    )
    mae  = mean_absolute_error(test["volume"], test["predicted_volume"])

    # Directional accuracy on event hours (hours_to_event < 4 and attendance > 0)
    event_hours = test[(test["hours_to_event"].abs() < 4) & (test["attendance"] > 0)].copy()
    if len(event_hours):
        dir_acc = (
            np.sign(event_hours["volume_lift"]) == np.sign(event_hours["predicted_lift"])
        ).mean()
    else:
        dir_acc = float("nan")

    print(f"  Test MAPE:               {mape:.2%}")
    print(f"  Test MAE:                {mae:.0f} vehicles/hr")
    print(f"  Event directional acc.:  {dir_acc:.2%}" if not np.isnan(dir_acc) else "  Event directional acc.:  n/a (no event hours in test)")

    # Feature importance
    fi = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)

    metrics = {
        "test_mape":           round(mape, 4),
        "test_mae_vehicles":   round(mae, 1),
        "event_directional_accuracy": round(dir_acc, 4) if not np.isnan(dir_acc) else None,
        "train_rows":          len(train),
        "test_rows":           len(test),
        "features":            FEATURES,
    }
    return model, test, fi, metrics


def main():
    print("Loading data...")
    traffic, events = load_data()

    print("Computing baselines...")
    traffic["hour"] = traffic["datetime"].dt.hour
    traffic["dow"]  = traffic["datetime"].dt.dayofweek
    baseline = compute_baseline(traffic)

    print("Building feature matrix...")
    features_df = build_feature_matrix(traffic, events, baseline)

    print("Training XGBoost model...")
    model, test_df, fi, metrics = train_and_evaluate(features_df)

    # Save outputs
    test_df.to_csv(OUTPUT_DIR / "volume_lift_predictions.csv", index=False)
    fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    with open(OUTPUT_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOutputs saved to outputs/")
    print(f"\nTop features by importance:")
    for _, r in fi.head(5).iterrows():
        print(f"  {r['feature']:25s}  {r['importance']:.3f}")

    print("\nRun visualization:  python src/visualize.py")


if __name__ == "__main__":
    main()
