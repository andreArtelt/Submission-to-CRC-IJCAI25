import numpy as np
import matplotlib.pyplot as plt
import momepy
import pandas as pd
import geopandas as gpd
def convert_to_numerical(value):
    # Convert path_type to numerical value
    if value == "walk":
        return 0.0
    elif value == "bike":
        return 1
    elif value == "walk_bike_connection":
        return 0.0
    elif value == "None":
        return 0.0
    else:
        # Check if value is a numeric type before calling np.isnan()
        if isinstance(value, (int, float)) and np.isnan(value):
            return 0.0
        else:
            return value

def get_attributes(G, attrs_variable_names):
    attributes = {}

    if G.is_multigraph():
        for u, v, k, data in G.edges(data=True, keys=True):
            edge_key = (u, v, k)
            attributes[edge_key] = {
                attr: convert_to_numerical(data.get(attr)) for attr in attrs_variable_names
            }
    else:
        for u, v, data in G.edges(data=True):
            edge_key = (u, v)
            attributes[edge_key] = {
                attr: convert_to_numerical(data.get(attr)) for attr in attrs_variable_names
            }

    return attributes
    

def remove_redundant_edges(df_route, attrs_variable_names):
    # Convert geometry to hashable WKT for duplicate detection
    df_handle = df_route.copy()
    df_handle['geometry_wkt'] = df_handle.geometry.apply(lambda g: g.wkt)
    
    # Check duplicates across all attributes + geometry
    cols = [c for c in df_handle.columns if c not in attrs_variable_names + ['geometry', 'geometry_wkt']]
    cols.append('geometry_wkt')  # Include hashed geometry
    
    # Keep first occurrence of duplicates
    df_dedup = df_handle.drop_duplicates(subset=cols).drop(columns='geometry_wkt')
    
    return df_dedup

def get_modified_edges_df(df_a, df_b, attrs_variable_names):
       # Check that geometries match across all rows
    if not (df_a['geometry'] == df_b['geometry']).all():
        raise ValueError("Edge geometry is not the same")

    # Safe comparison with NaN handling
    mask = pd.DataFrame({
        attr: ~((df_a[attr] == df_b[attr]) | (df_a[attr].isna() & df_b[attr].isna()))
        for attr in attrs_variable_names
    })

    differing_rows_idx = mask.any(axis=1).to_numpy().nonzero()[0]

    modified_edges = []

    for i in differing_rows_idx:
        for attr in attrs_variable_names:
            if mask.loc[i, attr]:
                modified_edges.append({
                    'length': df_a.loc[i, 'length'],
                    'geometry': df_a.loc[i, 'geometry'],
                    'attribute': attr,
                    'value_a': df_a.loc[i, attr],
                    'value_b': df_b.loc[i, attr]
                })

    result_gdf = gpd.GeoDataFrame(modified_edges, geometry='geometry', crs=df_a.crs)

    return result_gdf

def get_modified_edges_G(G_a, G_b, attrs_variable_names):
    """
    Compare two graphs and return a dictionary of GeoDataFrames containing modified edges from G_a,
    categorized by the attribute name that changed.
    
    Args:
        G_a: First networkx graph
        G_b: Second networkx graph
        attrs_variable_names: List of attribute names to compare
        
    Returns:
        Dictionary where keys are attribute names and values are GeoDataFrames containing
        the edges that were modified for that specific attribute
    """
    a_attributes = get_attributes(G_a, attrs_variable_names)
    b_attributes = get_attributes(G_b, attrs_variable_names)
    
    # Dictionary to store modified edges for each attribute
    modified_edges_by_attr = {attr_name: set() for attr_name in attrs_variable_names}
    
    for (a_uv, a_attrs), (b_uv, b_attrs) in zip(a_attributes.items(), b_attributes.items()):
        if a_uv != b_uv:
            print(f"Edge {a_uv} not in b")
            continue
            
        for attr_name in attrs_variable_names:
            if a_attrs.get(attr_name) != b_attrs.get(attr_name):
                modified_edges_by_attr[attr_name].add(a_uv)
    
    # Create dictionary of GeoDataFrames for each attribute
    gdf_by_attr = {attr_name: {} for attr_name in attrs_variable_names}
    for attr_name, edges in modified_edges_by_attr.items():
        if edges:  # Only create GeoDataFrame if there are modified edges
            # Create subgraph of G_a with only modified edges for this attribute
            G_modified =  G_b.edge_subgraph(edges).copy()
            # Convert to GeoDataFrame
            gdf_modified_edges = momepy.nx_to_gdf(G_modified, lines=True, points=False)
            gdf_by_attr[attr_name] = gdf_modified_edges
    
    return gdf_by_attr



