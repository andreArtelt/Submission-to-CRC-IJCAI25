from typing import Any
from shapely import from_wkt

class graphOperator():
    def __init__(self, map_constraint = None):
        if map_constraint is None:
            self.map_constraint = {
                "obstacle_free_width_float": {"bound":[0.6,2]},
                "curb_height_max": {"bound":[0,0.2]},
                "path_type": {"categorical_options":["walk","bike"], "bound":[0,1]},
            }
        else:
            self.map_constraint = map_constraint

        self.operators = [
            self.add_width,
            self.sub_width,
            self.add_curb_height,
            self.sub_curb_height,
            self.modify_path_type,
        ]

        self.operator_names = [
            'add_width',
            'sub_width',
            'add_curb_height',
            'sub_curb_height',
            'modify_path_type',
        ]

        self.operator_dict = {
            "add_width": self.add_width,
            "sub_width": self.sub_width,
            "add_curb_height": self.add_curb_height,
            "sub_curb_height": self.sub_curb_height,
            "modify_path_type": self.modify_path_type,
        }

    def get_numerical_bound_op(self, operator_name):
        if operator_name in ["add_width", "sub_width"]:
            return "obstacle_free_width_float", self.map_constraint["obstacle_free_width_float"]["bound"]
        elif operator_name in ["add_curb_height", "sub_curb_height"]:
            return "curb_height_max", self.map_constraint["curb_height_max"]["bound"]
        else:
            raise NotImplementedError(f"Numerical bound for {operator_name} not implemented")
    
    def numerical_modify(self, edge, df, attr_name, step, bound):
        #modify the edge with the attr_name
        min_bound = bound[0]
        max_bound = bound[1]
        value = edge[attr_name].iloc[0]
        if value + step <= max_bound and value + step >= min_bound:
            # Get the index of the edge in the original DataFrame
            edge_index = edge.index[0]
            # Update the value in the original DataFrame
            new_value = value + step
            # Update the original DataFrame with the modified value
            df.loc[edge_index, attr_name] = new_value
            return "success"
        else:
            return "failed: out of bound"
        
    def categorical_modify(self, edge, df, attr_name, option, categorical_options):
        if option in categorical_options:
            if edge[attr_name].iloc[0] != option:
                # Update the original DataFrame with the modified value
                edge_index = edge.index[0]
                df.loc[edge_index, attr_name] = option
                return "success"
            else:
                return "failed: already the same"
        else:
            return "failed: option not in categorical_options"

    def add_width(self, edge, df_original, step):
        #"obstacle_free_width_float"
        bound = self.map_constraint["obstacle_free_width_float"]["bound"]
        result = self.numerical_modify(edge, df_original, "obstacle_free_width_float", step, bound)
        return result

    def sub_width(self, edge, df_original, step):
        #"obstacle_free_width_float"
        bound = self.map_constraint["obstacle_free_width_float"]["bound"]
        result = self.numerical_modify(edge, df_original, "obstacle_free_width_float", -step, bound)
        return result

    def add_curb_height(self, edge, df_original, step):
        #"curb_height_max"
        if not edge["crossing_type"].iloc[0] == "curb_height":
            return "failed: not a curb_height_max crossing"
        bound = self.map_constraint["curb_height_max"]["bound"]
        result = self.numerical_modify(edge, df_original, "curb_height_max", step, bound)
        return result

    def sub_curb_height(self, edge, df_original, step):
        #"curb_height_max"
        if not edge["crossing_type"].iloc[0] == "curb_height":
            return "failed: not a curb_height_max crossing"
        bound = self.map_constraint["curb_height_max"]["bound"]
        result = self.numerical_modify(edge, df_original, "curb_height_max", -step, bound)
        return result

    def modify_path_type(self, edge, df_original, option):
        #"path_type": only allow to change walk or bike

        if edge["path_type"].iloc[0] != "walk" and edge["path_type"].iloc[0] != "bike":
            return "failed: path_type not allowed to be changed"
        
        categorical_options = self.map_constraint["path_type"]["categorical_options"]
        result = self.categorical_modify(edge, df_original, "path_type", option, categorical_options)
        return result
    
    def get_categorical_options(self, attr_name):
        return self.map_constraint[attr_name]["categorical_options"]
    
    def get_numerical_bound(self, attr_name):
        return self.map_constraint[attr_name]["bound"]
    

def pertub_with_op_list(graph_operator, op_list, df):
    df_perturbed = df.copy()
    op_list_perturbed = []
    for op in op_list:
        operator_name = op[0]
        edge_index = op[1][0]
        edge_geom = op[1][1]
        step = op[2]
        result_op = op[3]
        operator = graph_operator.operator_dict[operator_name]
        target_edge = df_perturbed.iloc[[edge_index]]
        if type(edge_geom) == str:
            compared_geom = from_wkt(edge_geom)
        else:
            compared_geom = edge_geom
        assert target_edge.geometry.iloc[0].equals(compared_geom), "Edge geometry does not match"

        result = operator(target_edge, df_perturbed, step)
        if result_op != result:
            print("result", result)
            print(f"Failed to apply operator {operator_name} with {step} to edge {edge_index}")
            if operator_name != "modify_path_type":
                print(target_edge["obstacle_free_width_float"], target_edge["path_type"])
                after_perturb = target_edge["obstacle_free_width_float"] + step
                print(after_perturb)
            else:
                print(target_edge["path_type"], step)
                
        op_list_perturbed.append((operator_name, (edge_index, edge_geom), step, result))
    return df_perturbed, op_list_perturbed