import numpy as np
from utils.helper import  remove_redundant_edges
import pandas as pd
from shapely import to_wkt, from_wkt


def common_edges_similarity_route_df_weighted(df_route_a, df_route_b, attrs_variable_names):

    df_a = remove_redundant_edges(df_route_a.copy(), attrs_variable_names)
    df_b = remove_redundant_edges(df_route_b.copy(), attrs_variable_names)
    df_a['geometry_wkt'] = df_a['geometry'].apply(lambda g: g.wkt)
    df_b['geometry_wkt'] = df_b['geometry'].apply(lambda g: g.wkt)

    # Inner join on geometry_wkt to find common edges
    merged = pd.merge(df_a[['geometry_wkt', 'mm_len']], df_b[['geometry_wkt', 'mm_len']], on=['geometry_wkt', 'mm_len'])

    # Sum of lengths of common edges from df_route_a
    common_edges_sum = merged['mm_len'].sum()

    # Total edge length sum from both dataframes
    total_edges_sum = sum(df_a['mm_len']) + sum(df_b['mm_len'])

    similarity = 2*common_edges_sum / total_edges_sum

    return round(similarity, 8)


def get_edge_attributes_df(df, attrs_variable_names):
    df_handle = df.copy()
    df_handle['geometry_wkt'] = df_handle.geometry.apply(lambda g: to_wkt(g, rounding_precision=-1, trim=False))
    columns_to_keep = attrs_variable_names + ['geometry_wkt']
    return df_handle[columns_to_keep]
 
def subtract_with_categorical(df_a, df_b, attrs_variable_names, ):
    result = pd.DataFrame(index=df_a.index)
    
    for col in attrs_variable_names:
        if col == "path_type":
            # If values are the same: 0; else: take from df_b
            result[col] = df_b[col].where(df_a[col] != df_b[col], other=0)
        else:
            # Standard numeric subtraction
            result[col] = df_b[col] - df_a[col]
    result['geometry_wkt'] = df_a['geometry_wkt']
    nonzero_mask = result[attrs_variable_names].apply(
        lambda row: any(val not in [0, None] and pd.notna(val) for val in row), axis=1
    )
    result = result[nonzero_mask]
    return result

def get_operator_list(sub_result,attrs_variable_names):
    op_list = []
    for index, row in sub_result.iterrows():
        for attr in attrs_variable_names:
            if row[attr] not in [0, None] and pd.notna(row[attr]):
                if attr == "path_type":
                    op_name = "modify_path_type"
                elif attr == "curb_height_max":
                    op_name = "add_curb_height"
                elif attr == "obstacle_free_width_float":
                    op_name = "add_width"
                op_list.append((op_name, (index, from_wkt(row['geometry_wkt'])), row[attr], "success"))
    return op_list

# sub_op_list = get_operator_list(result,attrs_variable_names)

def get_virtual_op_list(df_a, df_b, attrs_variable_names):
    df_a_G = get_edge_attributes_df(df_a, attrs_variable_names)
    df_b_G = get_edge_attributes_df(df_b, attrs_variable_names)
    result = subtract_with_categorical(df_a_G, df_b_G, attrs_variable_names)
    sub_op_list = get_operator_list(result,attrs_variable_names)
    return sub_op_list