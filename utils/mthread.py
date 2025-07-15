
import numpy as np
from utils.dataparser import  create_network_graph, handle_weight,handle_weight_with_recovery

from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list

def generate_neighbor_p( df_G, router, graph_operator, origin_node, dest_node, args, user_model ):
        heuristic_f = args["heuristic_f"]
        n_perturbation = args["n_perturbation"]
        operator_p = args["operator_p"]

        while True:
            df_perturbed = df_G.copy()

            operator_names = graph_operator.operator_names
            op_list = []
            for i in range(n_perturbation):
                # randomly choose an attribute to perturb
                edge = df_perturbed.sample(1)
                edge_index = edge.index[0]
                # Select a random operator
                operator_name = np.random.choice(operator_names, p=operator_p)
                operator = graph_operator.operator_dict[operator_name]
                # Apply the operator with appropriate parameters
                if operator == graph_operator.modify_path_type:
                    # For path_type, we need to provide an option
                    edge_attr = "path_type"
                    options = graph_operator.get_categorical_options(edge_attr)
                    step = np.random.choice(options)
                    result = operator(edge, df_perturbed, step)
                else:
                    # pass the edge, df_modified, step
                    attr_name, bound = graph_operator.get_numerical_bound_op(operator_name)
                    edge_attr = edge[attr_name].iloc[0]
                    step = np.random.choice([bound[1]-edge_attr, bound[0]-edge_attr])
                    result = operator(edge, df_perturbed, step)
                op_list.append((operator_name, (edge_index, edge["geometry"]), step, result))

            df_perturbed = handle_weight_with_recovery(df_perturbed, user_model)
            G_perturbed, G_perturbed = create_network_graph(df_perturbed)

            # there could be no route
            try:
                path_p, G_path_p, df_path_p = router.get_route(G_perturbed, origin_node, dest_node, heuristic_f=heuristic_f)
            except Exception as e:
                continue
            break
        return (df_perturbed, G_perturbed),(df_path_p, G_path_p), op_list

def parallel_generate_neighbor(df, router, graph_operator, origin_node, dest_node, df_foil_route, args, user_model):
        (df_perturbed_i, G_perturbed_i),(df_path_i, G_path_i), op_list_perturbed = \
            generate_neighbor_p(df, router, graph_operator, origin_node, dest_node, args, user_model)

        sub_op_list = get_virtual_op_list(df, df_perturbed_i, args["attrs_variable_names"])
        graph_error = len([op for op in sub_op_list if op[3] == "success"])


        route_similarity = common_edges_similarity_route_df_weighted(df_path_i, df_foil_route, args["attrs_variable_names"])
        route_error = 1-route_similarity
        assert route_error >= 0, "Route error is negative"
        return (df_perturbed_i, G_perturbed_i, graph_error), (df_path_i, G_path_i, route_error), (op_list_perturbed, sub_op_list)