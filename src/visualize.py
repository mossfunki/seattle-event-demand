"""
visualize.py
Generates an interactive Folium map showing predicted demand lift by road corridor.
Saves: outputs/seattle_demand_map.html
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import folium
    from folium.plugins import HeatMap
except ImportError:
    raise SystemExit("Run:  pip install folium")

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

STATION_COORDS = {
    110: (47.578,  -122.327, "I-90 at 4th Ave S (SODO)"),
    102: (47.624,  -122.330, "I-5 NB at Mercer St"),
    108: (47.572,  -122.326, "I-5 SB at S Spokane St"),
    215: (47.608,  -122.342, "SR-99 at Western Ave"),
    320: (47.498,  -122.194, "I-405 at Renton Ave"),
}

VENUE_LOCATIONS = {
    "Lumen Field":          (47.5952, -122.3316, 68000,  "#ef4444"),
    "T-Mobile Park":        (47.5914, -122.3324, 30000,  "#3b82f6"),
    "Climate Pledge Arena": (47.6220, -122.3542, 17000,  "#8b5cf6"),
    "Convention Center":    (47.6101, -122.3326, 70000,  "#f59e0b"),
    "Lake Washington":      (47.5950, -122.2830, 50000,  "#22c55e"),
}


def lift_color(lift_pct: float) -> str:
    if lift_pct < 0.05:  return "#22c55e"
    if lift_pct < 0.15:  return "#f59e0b"
    if lift_pct < 0.30:  return "#ef4444"
    return "#991b1b"


def lift_label(lift_pct: float) -> str:
    if lift_pct < 0.05:  return "Normal"
    if lift_pct < 0.15:  return "Elevated"
    if lift_pct < 0.30:  return "High"
    return "Critical"


def make_map(predictions: pd.DataFrame, events: pd.DataFrame) -> folium.Map:
    m = folium.Map(location=[47.605, -122.330], zoom_start=12,
                   tiles="CartoDB dark_matter")

    # ── Venue markers ─────────────────────────────────────────────────────────
    venue_group = folium.FeatureGroup(name="Event Venues", show=True)
    for venue, (lat, lon, att, color) in VENUE_LOCATIONS.items():
        folium.CircleMarker(
            location=[lat, lon],
            radius=14,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.35,
            weight=2,
            popup=folium.Popup(f"<b>{venue}</b><br>Capacity: {att:,}", max_width=200),
            tooltip=venue,
        ).add_to(venue_group)
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-family:monospace;font-size:10px;color:{color};font-weight:600;white-space:nowrap;">{venue.split()[0]}</div>',
                icon_size=(100, 20), icon_anchor=(0, 10)
            )
        ).add_to(venue_group)
    venue_group.add_to(m)

    # ── Traffic station markers (colored by predicted lift) ────────────────────
    station_group = folium.FeatureGroup(name="Traffic Corridors", show=True)
    for station_id, (lat, lon, name) in STATION_COORDS.items():
        # Average predicted lift for this station across all event hours
        station_preds = predictions[predictions["station_id"] == station_id]
        event_preds   = station_preds[station_preds["att_large"].fillna(0) + station_preds["att_medium"].fillna(0) > 0]

        avg_lift = event_preds["predicted_lift"].mean() if len(event_preds) else 0.0
        color    = lift_color(avg_lift)
        label    = lift_label(avg_lift)
        radius   = 10 + min(avg_lift * 80, 20)

        popup_html = f"""
        <b>{name}</b><br>
        Avg event lift: <b>{avg_lift:.1%}</b><br>
        Status: <b style="color:{color}">{label}</b><br>
        Observations: {len(event_preds)}
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=2,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{name}: {avg_lift:.1%} event lift",
        ).add_to(station_group)
    station_group.add_to(m)

    # ── Heat map of predicted volume lift ─────────────────────────────────────
    heatmap_data = []
    for station_id, (lat, lon, name) in STATION_COORDS.items():
        s = predictions[
            (predictions["station_id"] == station_id) &
            (predictions["predicted_lift"].notna())
        ]
        if len(s):
            lift = s["predicted_lift"].clip(0).mean()
            heatmap_data.append([lat, lon, float(lift)])

    if heatmap_data:
        heat_group = folium.FeatureGroup(name="Volume Lift Heatmap", show=False)
        HeatMap(heatmap_data, min_opacity=0.3, radius=30, blur=20,
                gradient={0.2: "#22c55e", 0.5: "#f59e0b", 0.8: "#ef4444", 1.0: "#991b1b"}
                ).add_to(heat_group)
        heat_group.add_to(m)

    # ── Event markers ─────────────────────────────────────────────────────────
    event_group = folium.FeatureGroup(name="Upcoming Events", show=True)
    for _, ev in events.iterrows():
        venue = ev.get("venue", "")
        if venue in VENUE_LOCATIONS:
            lat, lon, _, color = VENUE_LOCATIONS[venue]
            folium.Marker(
                location=[lat + 0.003, lon],
                icon=folium.DivIcon(
                    html=f'<div style="background:{color};color:#fff;padding:2px 6px;font-size:9px;border-radius:2px;font-family:monospace;white-space:nowrap;">{ev["event_name"]} &mdash; {ev["attendance"]:,}</div>',
                    icon_size=(200, 18), icon_anchor=(0, 9)
                )
            ).add_to(event_group)
    event_group.add_to(m)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;background:rgba(16,20,30,0.92);
                border:1px solid rgba(255,255,255,0.15);border-radius:4px;
                padding:12px 16px;font-family:monospace;font-size:11px;color:#e6e3dc;z-index:9999;">
      <div style="font-weight:600;margin-bottom:8px;color:#c8a152;letter-spacing:0.08em;">VOLUME LIFT</div>
      <div><span style="color:#22c55e">&#9679;</span> &lt;5%  — Normal</div>
      <div><span style="color:#f59e0b">&#9679;</span> 5–15% — Elevated</div>
      <div><span style="color:#ef4444">&#9679;</span> 15–30% — High</div>
      <div><span style="color:#991b1b">&#9679;</span> &gt;30% — Critical</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    print("Loading prediction data...")
    predictions = pd.read_csv(OUTPUT_DIR / "volume_lift_predictions.csv", parse_dates=["datetime"])
    events      = pd.read_csv(DATA_DIR / "events.csv", parse_dates=["date"])

    print("Building interactive map...")
    m = make_map(predictions, events)

    out_path = OUTPUT_DIR / "seattle_demand_map.html"
    m.save(str(out_path))
    print(f"Map saved → {out_path}")
    print("Open in browser: open outputs/seattle_demand_map.html")


if __name__ == "__main__":
    main()
