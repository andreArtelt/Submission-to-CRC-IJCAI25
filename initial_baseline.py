"""
Initial baseline method for computing an approximate solution taken from ls_demo.ipynb
"""
import json
import pandas as pd
import geopandas as gpd

# Local or project-specific imports
from utils.helper import get_modified_edges_df
from router import Router
from utils.graph_op import graphOperator
from utils.dataparser import  create_network_graph, handle_weight, handle_weight_with_recovery
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list

import multiprocessing as mp
from copy import deepcopy 
from shapely import wkt
from utils.mthread import generate_neighbor_p, parallel_generate_neighbor
from utils.dataparser import convert
from shapely import to_wkt


class LS():
    def __init__(self, df, path_foil, df_path_foil, gdf_coords_loaded, user_model, meta_map, attrs_variable_names):
        self.df = df
        self.path_foil = path_foil
        self.df_path_foil = df_path_foil
        self.gdf_coords_loaded = gdf_coords_loaded
        self.user_model = user_model
        self.meta_map = meta_map
        self.attrs_variable_names = attrs_variable_names

    def reset(self):
        self.heuristic = "dijkstra"
        self.heuristic_f = "my_weight"
        self.jobs = -1
        if self.jobs > 1:
            self.pool = mp.Pool(processes=self.jobs)
        else:
            self.pool = None

        df_copy = deepcopy(self.df)
        df_copy = handle_weight(df_copy, self.user_model)
        _, self.G = create_network_graph(df_copy)

        self.router_h= Router(heuristic=self.heuristic, CRS=self.meta_map["CRS"], CRS_map=self.meta_map["CRS_map"])
        self.graph_operator = graphOperator()
        self.origin_node, self.dest_node, self.origin_node_loc, self.dest_node_loc, self.gdf_coords = \
            self.router_h.set_o_d_coords(self.G, self.gdf_coords_loaded)

        self.path_fact, self.G_path_fact, self.df_path_fact = \
            self.router_h.get_route(self.G, self.origin_node, self.dest_node, self.heuristic_f)

    def generate_neighbor(self, df):
        (df_perturbed_i, G_perturbed_i), (df_path_i, G_path_i), op_list_perturbed = \
            generate_neighbor_p(df, self.router_h, self.graph_operator, self.origin_node, self.dest_node,
                                {"attrs_variable_names": self.attrs_variable_names, "operator_p": [0.15, 0.15, 0.15, 0.15, 0.4],
                                 "n_perturbation": 50, "heuristic_f": self.heuristic_f},
                                self.user_model)
        if df_perturbed_i is None:
            return (None, None, 0), (None, None, 0), op_list_perturbed

        sub_op_list = get_virtual_op_list(self.df, df_perturbed_i, self.attrs_variable_names)
        graph_error = len([op for op in sub_op_list if op[3] == "success"])

        route_error = 1-common_edges_similarity_route_df_weighted(df_path_i, self.df_path_foil, self.attrs_variable_names)
        return (df_perturbed_i, G_perturbed_i, graph_error), (df_path_i, G_path_i, route_error), (op_list_perturbed, sub_op_list)

    def generate_population(self, df, pop_num):
        pop = []
        if self.jobs > 1:
            jobs = [self.pool.apply_async(parallel_generate_neighbor, (df, self.router_h, self.graph_operator,
                                                                       self.origin_node, self.dest_node, self.df_path_foil,
                                                                       {"attrs_variable_names": self.attrs_variable_names,
                                                                        "operator_p": [0.15, 0.15, 0.15, 0.15, 0.4], "n_perturbation": 50,
                                                                        "heuristic_f": self.heuristic_f}, self.user_model, ))
                                                                        for _ in range(pop_num)]
            for idx, j in enumerate(jobs):
                try:
                    (df_perturbed_i, G_perturbed_i, graph_error), (df_path_i, G_path_i, route_error), op_lists = j.get()
                except Exception as e:
                    print(e)
                    (df_perturbed_i, G_perturbed_i, graph_error), (df_path_i, G_path_i, route_error), op_lists = (0, 0, 0), (0, 0, 0), []

                pop.append(((df_perturbed_i, G_perturbed_i, graph_error), (df_path_i, G_path_i, route_error), op_lists))
        else:
            for _ in range(pop_num):
                pop.append((self.generate_neighbor(df)))
        return pop

    def get_perturbed_edges(self, df_perturbed):
        modified_edges_df = get_modified_edges_df(self.df, df_perturbed, self.attrs_variable_names)
        return modified_edges_df


def read_data(args):

    basic_network_path = args['basic_network_path']
    foil_json_path = args['foil_json_path']
    df_path_foil_path = args['df_path_foil_path']
    gdf_coords_path = args['gdf_coords_path']

    df = gpd.read_file(basic_network_path)
    with open(foil_json_path, 'r') as f:
        path_foil = json.load(f)

    df_path_foil = gpd.read_file(df_path_foil_path)
    gdf_coords_loaded = pd.read_csv(gdf_coords_path, sep=';')

    gdf_coords_loaded['geometry'] = gdf_coords_loaded['geometry'].apply(wkt.loads)
    gdf_coords_loaded = gpd.GeoDataFrame(gdf_coords_loaded, geometry='geometry')

    return df, path_foil, df_path_foil, gdf_coords_loaded


def run_baseline_method(df, path_foil, df_path_foil, gdf_coords_loaded, user_model, meta_map, attrs_variable_names):
    best_weighted_error = 1000000
    best_graph_error = 1000
    best_route_error = 1000
    gen_num = 10
    lagrangian_lambda = 2000
    route_error_delta = user_model["route_error_threshold"]

    ls = LS(df, path_foil, df_path_foil, gdf_coords_loaded, user_model, meta_map, attrs_variable_names)
    ls.reset()

    best_df = [ls.df.copy()]
    best_route = [None]
    best_log = []
    gen_log = []

    # compare fact and foil route
    fact_path, G_fact_path, df_fact_path  = ls.router_h.get_route(ls.G, ls.origin_node, ls.dest_node, ls.heuristic_f)
    route_similarity = common_edges_similarity_route_df_weighted(df_fact_path, ls.df_path_foil, ls.attrs_variable_names)
    print("error of fact route and foil route", 1-route_similarity)

    def penalty_function(error, error_delta):
        error = max(0, error - error_delta)
        return error

    for gen in range(gen_num):
        pop = ls.generate_population(best_df[0], 10)
        sorted_pop = sorted(pop, key=lambda x: x[0][2]+lagrangian_lambda*penalty_function(x[1][2], route_error_delta))
        print(gen,"graph error:", sorted_pop[0][0][2], "route error:",sorted_pop[0][1][2],
              "penalty:", penalty_function(sorted_pop[0][1][2], route_error_delta))
        current_best_weighted_error = sorted_pop[0][0][2]+lagrangian_lambda*penalty_function(sorted_pop[0][1][2], route_error_delta)
        if current_best_weighted_error < best_weighted_error:
            best_df = sorted_pop[0][0]
            best_route = sorted_pop[0][1]
            best_weighted_error = current_best_weighted_error
            best_graph_error = sorted_pop[0][0][2]
            best_route_error = sorted_pop[0][1][2]
        best_log.append((gen, best_df, best_weighted_error, best_graph_error,best_route_error))
        gen_log.append((gen, sorted_pop[0][0][0], current_best_weighted_error, sorted_pop[0][0][2], sorted_pop[0][1][2]))

    # Return final results
    v_op_list = get_virtual_op_list(ls.df, best_df[0], attrs_variable_names)
    available_op = [(op[0], (convert(op[1][0]), to_wkt(op[1][1], rounding_precision=-1, trim=False)),
                     convert(op[2]), op[3]) for op in v_op_list if op[3] == "success"]

    map_df = best_df[0]
    op_list = available_op

    return map_df, op_list
