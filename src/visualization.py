import folium
from folium.plugins import HeatMap


def plot_interactive_hotspot_map(hotspot_df, lat_col='lat_bin', lon_col='lon_bin', value_col='mean_radiation', is_above_col='is_above_who', map_center=None, zoom_start=2, save_path=None):
    """
    Plot interactive map of radiation hotspots using folium:
    - Grid cells above WHO threshold are marked in red, others in blue.
    - Can save to HTML file if save_path is provided.
    """
    if map_center is None:
        # Calculate map center
        map_center = [hotspot_df[lat_col].mean(), hotspot_df[lon_col].mean()]
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles='cartodbpositron')
    # Add background heatmap
    heat_data = hotspot_df[[lat_col, lon_col, value_col]].values.tolist()
    HeatMap(heat_data, radius=12, blur=18, min_opacity=0.3, max_zoom=1).add_to(m)
    # Mark hotspots above threshold
    for _, row in hotspot_df.iterrows():
        color = 'red' if row[is_above_col] else 'blue'
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Radiation: {row[value_col]:.3f} μSv/h"
        ).add_to(m)
    if save_path:
        m.save(save_path)
    return m 