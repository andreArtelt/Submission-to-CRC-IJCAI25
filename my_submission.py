import os
import argparse
import json
import time


from copy import deepcopy
import pandas as pd
import geopandas as gpd
from shapely import wkt, to_wkt
from router import Router
from utils.graph_op import graphOperator
from utils.dataparser import create_network_graph, handle_weight, convert
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
from initial_baseline import run_baseline_method


def load_scenario(basic_network_path: str, foil_json_path: str, df_path_foil_path: str,
                  gdf_coords_path: str, meta_data_path: str):
    # Read meta data
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    # Profile settings
    user_model = meta_data["user_model"]
    meta_map = meta_data["map"]

    attrs_variable_names = user_model["attrs_variable_names"]
    #route_error_delta = user_model["route_error_threshold"]

    # Load data frames
    df = gpd.read_file(basic_network_path)
    with open(foil_json_path, 'r') as f:
        path_foil = json.load(f)

    if df_path_foil_path is not None:
        df_path_foil = gpd.read_file(df_path_foil_path)
    else:
        df_path_foil = None
    gdf_coords_loaded = pd.read_csv(gdf_coords_path, sep=';')

    gdf_coords_loaded['geometry'] = gdf_coords_loaded['geometry'].apply(wkt.loads)
    gdf_coords_loaded = gpd.GeoDataFrame(gdf_coords_loaded, geometry='geometry')

    path_foil = list(map(lambda edge: (edge[0], edge[1]), path_foil))

    return df, path_foil, df_path_foil, gdf_coords_loaded, user_model, meta_map, \
        attrs_variable_names


def build_graph(df, gdf_coords_loaded, user_model, meta_map):
    df_copy = deepcopy(df)
    df_copy = handle_weight(df_copy, user_model)
    _, G = create_network_graph(df_copy)

    router_h = Router(heuristic="dijkstra", CRS=meta_map["CRS"],
                      CRS_map=meta_map["CRS_map"])
    origin_node, dest_node, _, _, _ = router_h.set_o_d_coords(G, gdf_coords_loaded)

    _, _, df_path_fact = router_h.get_route(G, origin_node, dest_node, "dijkstra")

    return G, origin_node, dest_node, router_h, df_path_fact


def compute_route(router_h, graph, start_node, dest_node):
    path_fact, _, _ = router_h.get_route(graph, start_node, dest_node, "dijkstra")

    return path_fact


def get_df_idx_of_edge(df, node_a, node_b) -> int:
    idx = []
    for i, edge in enumerate(df["geometry"].to_list()):
        if node_a in edge.coords and node_b in edge.coords:
            idx.append(i)

    return idx[0]


def recompute_path(df, gdf_coords_loaded, user_model, meta_map, attrs_variable_names, df_path_foil):
    G, origin_node, dest_node, router_h, df_path_fact = build_graph(df, gdf_coords_loaded,
                                                                    user_model, meta_map)
    path_fact = compute_route(router_h, G, origin_node, dest_node)
    route_error = 1 - common_edges_similarity_route_df_weighted(df_path_fact, df_path_foil,
                                                                attrs_variable_names)

    return path_fact, route_error


def get_results(args):
    # Load scenario
    df, path_foil, df_path_foil, gdf_coords_loaded, user_model, meta_map, attrs_variable_names = \
        load_scenario(args.basic_network_path, args.foil_json_path, args.df_path_foil_path,
                      args.gdf_coords_path, args.meta_data_path)

    # Compute initial path and compare it to the requested path
    path_fact, route_error = recompute_path(df, gdf_coords_loaded, user_model, meta_map,
                                            attrs_variable_names, df_path_foil)

    # Algorithm
    # 0. Generate a counterfactual graph/map that yields the foil route
    df_orig = df.copy()

    # 1. Make sure all edges from the foil path are included and the the path types are
    # according to the user's preference
    for i in range(1, len(path_foil)):
        graph_operator = graphOperator()
        idx = get_df_idx_of_edge(df, path_foil[i-1], path_foil[i])
        edge = df.iloc[idx]

        if edge["curb_height_max"].item() > user_model["max_curb_height"]:
            if graph_operator.sub_curb_height(df.iloc[idx:idx+1], df,
                                              edge["curb_height_max"].item()  - \
                                                user_model["max_curb_height"] + 0.00001) != "success":
                print("Can not change 'curb_height_max'")

        if edge["obstacle_free_width_float"].item() < user_model["min_sidewalk_width"]:
            if graph_operator.add_width(df.iloc[idx:idx+1], df,
                                        user_model["min_sidewalk_width"] - \
                                            edge["obstacle_free_width_float"].item() + 0.00001) != "success":
                print("Can not change 'obstacle_free_width_float'")

        if edge["crossing_type"] is not None:
            # Can not be modified!
            pass

        if edge["path_type"] != "walk_bike_connection":
            if edge["path_type"] != user_model["walk_bike_preference"]:
                if graph_operator.modify_path_type(df.iloc[idx:idx+1], df,
                                                   user_model["walk_bike_preference"]) != "success":
                    print("Can not change 'path_type'")

    # Recompute shortest path and compare it to the requested path
    path_fact, route_error = recompute_path(df, gdf_coords_loaded, user_model, meta_map,
                                            attrs_variable_names, df_path_foil)

    n_max_itr = 100000  # TODO: magic number
    for _ in range(n_max_itr):  # Repeat with loop until path_fact = path_foil
        # 2. Remove all edges that lead to a different shortest path by changing
        # curb_height_max or obstacle_free_width_float
        change_flag = False
        for i in range(1, len(path_fact)):
            if path_fact[i] != path_foil[i]:
                graph_operator = graphOperator()
                idx = get_df_idx_of_edge(df, path_fact[i-1], path_fact[i])
                edge = df.iloc[idx]

                if edge["curb_height_max"].item() < user_model["max_curb_height"]:
                    cur_curb_height_max = edge["curb_height_max"].item()
                    step = user_model["max_curb_height"] - \
                        cur_curb_height_max + 0.00001

                    bounds = graph_operator.map_constraint["curb_height_max"]["bound"]
                    step = min(step, cur_curb_height_max - bounds[1])
                    if step <= 1e-10:
                        # Exact solution does not exist
                        # Fall back to baseline method for computing an approximate solution
                        print("Fallback to baseline method")
                        return run_baseline_method(df, path_foil, df_path_foil, gdf_coords_loaded,
                                                   user_model, meta_map, attrs_variable_names)

                    if graph_operator.add_curb_height(df.iloc[idx:idx+1], df, step) != "success":
                        print("Error: Can not chage 'curb_height_max'")
                    else:
                        change_flag = True
                        break

                if edge["obstacle_free_width_float"].item() >= user_model["min_sidewalk_width"]:
                    cur_obstacle_free_width_float = edge["obstacle_free_width_float"].item()
                    step = cur_obstacle_free_width_float - \
                        user_model["min_sidewalk_width"] + 0.00001

                    bounds = graph_operator.map_constraint["obstacle_free_width_float"]["bound"]
                    step = min(step, cur_obstacle_free_width_float - bounds[0])
                    if step <= 1e-10:
                        # Exact solution does not exist
                        # Fall back to baseline method for computing an approximate solution
                        print("Fallback to baseline method")
                        return run_baseline_method(df, path_foil, df_path_foil, gdf_coords_loaded,
                                                   user_model, meta_map, attrs_variable_names)

                    if graph_operator.sub_width(df.iloc[idx:idx+1], df, step) != "success":
                        print("Error: Can not change 'obstacle_free_width_float'")
                    else:
                        change_flag = True
                        break

        # Recompute shortest path and compare it to the requested path
        path_fact, route_error = recompute_path(df, gdf_coords_loaded, user_model, meta_map,
                                                attrs_variable_names, df_path_foil)

        print(route_error)
        if route_error == 0:  # Stop if requested path have been reached
            break

        if change_flag is False:    # Stop if no more changes can be made => no exact solution exists
            # Exact solution does not exist
            # Fall back to baseline method for computing an approximate solution
            print("Fallback to baseline method")
            return run_baseline_method(df, path_foil, df_path_foil, gdf_coords_loaded, user_model,
                                       meta_map, attrs_variable_names)

    # Get final graph operations
    v_op_list = get_virtual_op_list(df_orig, df, attrs_variable_names)
    available_op = [(op[0], (convert(op[1][0]), to_wkt(op[1][1], rounding_precision=-1, trim=False)),
                     convert(op[2]), op[3]) for op in v_op_list if op[3] == "success"]

    # Return results
    map_df = df
    op_list = available_op

    return map_df, op_list


def store_results(output_path, map_df, op_list):

    map_df_path = os.path.join(output_path, "map_df.gpkg")
    op_list_path = os.path.join(output_path, "op_list.json")

    map_df.to_file(map_df_path, driver='GPKG')
    with open(op_list_path, 'w') as f:
        json.dump(op_list, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_data_path", type=str, required=True)
    parser.add_argument("--basic_network_path", type=str, required=True)
    parser.add_argument("--foil_json_path", type=str, required=True)
    parser.add_argument("--df_path_foil_path", type=str, required=True)
    parser.add_argument("--gdf_coords_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    map_df, op_list = get_results(args)
    store_results(args.output_path, map_df, op_list)
    print(f"Total time elapsed: {time.time() - start}s")
