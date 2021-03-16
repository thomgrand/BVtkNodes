'''
This file is responsible for helping in animating properties until
https://developer.blender.org/T66392
is resolved.
'''
from ..core import *
from ..core import l
import bpy
import numpy as np

invalid_frame = int(-1e6)
invalid_vtk_time = -4.37e6

class AnimationHelper():

    current_frame = invalid_frame
    vtk_time = invalid_vtk_time
    def setup(self):
        self.current_frame = int(-1e6)
        self.animated_properties = None

    def get_animated_property_list(self, recalc=False):
        if recalc:
            animated_properties = []
            for f_curve in self.f_curves:
                if f_curve.prop_path in self.animated_properties:
                    animated_properties.append(f_curve.data_path)
                    #self.animated_values.append(f_curve.evaluate(current_frame))

        else:
            animated_properties = self.animated_properties #, self.animated_values

        return animated_properties

    def update_animated_properties(self, scene):
        current_frame = scene.frame_current
        #if not hasattr(self, 'current_frame') or self.current_frame != current_frame:
        #    self.current_frame = current_frame
        AnimationHelper.current_frame = current_frame
        AnimationHelper.vtk_time = current_frame * self.vtk_speed
            
        self.animated_properties = []
        self.interpolation_modes = []
        self.animated_values = {}
        for node_group in bpy.data.node_groups.values():
            node_tree_name = node_group.name
            for key, val in bpy.data.actions.items():

                #Skip unrelated node trees
                if node_tree_name + "Action" in key:
                    self.f_curves = bpy.data.actions["NodeTreeAction"].fcurves
                    #updated_prop_paths = {}
                    updated_nodes = set()

                    for f_curve in self.f_curves:
                        new_val = f_curve.evaluate(current_frame)
                        prop_path = f_curve.data_path
                        delimiter_index = prop_path.rindex(".")
                        node_name = prop_path[7:delimiter_index-2]
                        attribute_name = prop_path[delimiter_index+1:]
                        arr_ind = f_curve.array_index
                        try:
                            try:
                                current_val = eval(f_curve.data_path + "[{}]".format(arr_ind), 
                                        {"nodes": node_group.nodes})
                            except TypeError as err:
                                if arr_ind == 0: #May be a scalar
                                    current_val = eval(f_curve.data_path, {"nodes": node_group.nodes})
                                else:
                                    l.error("Could not load property " + prop_path 
                                                + ". Error: " + str(err))
                                    continue
                            
                        except Exception as ex:
                            l.error("Could not update property " + prop_path + ". Error: " + str(ex))
                            continue

                        interpolation_modes = [kf.interpolation for kf in f_curve.keyframe_points]
                        first_mode = interpolation_modes[0]
                        single_mode = all(x == first_mode for x in interpolation_modes)
                        self.interpolation_modes.append(first_mode if single_mode else "Mixed")

                        if prop_path not in self.animated_properties:
                            self.animated_properties.append(prop_path)
                            old_vals = (self.animated_values[prop_path] if prop_path in self.animated_values else ())
                            self.animated_values[prop_path] = old_vals + (new_val,)

                        #No update necessary if the value hasn't changed
                        if np.isclose(current_val, new_val):
                            continue

                        #Update the property with the new value
                        try:
                            try:
                                exec(f_curve.data_path + "[{}] = {}".format(arr_ind, new_val), 
                                        {"nodes": node_group.nodes})
                            except TypeError as err:
                                if arr_ind == 0: #May be a scalar update
                                    exec(f_curve.data_path + " = {}".format(new_val), 
                                        {"nodes": node_group.nodes})
                                else:
                                    l.error("Could not update property " + prop_path 
                                                + ". Error: " + str(err))

                            updated_node = eval(prop_path[:delimiter_index], {"nodes": node_group.nodes})
                            updated_nodes = updated_nodes.union(set([updated_node]))
                            #if prop_path in updated_prop_paths:
                            #    updated_prop_paths[prop_path] += (new_val,)
                            #else:
                            #    updated_prop_paths[prop_path] = (new_val,)
                        except Exception as ex:
                            l.error("Could not update property " + prop_path + ". Error: " + str(ex))

                    #This helps in better displaying vectors by concatenating into a list of tuples
                    self.animated_values = [self.animated_values[prop_path] for prop_path in self.animated_properties]
                    return updated_nodes
                    
                    


#  [kf.co for kf in bpy.data.actions["NodeTreeAction"].fcurves[2].keyframe_points]
# Remove:
# tmp =  bpy.data.actions["NodeTreeAction"].fcurves[0] 
# bpy.data.actions["NodeTreeAction"].fcurves.remove(tmp)
# Evaluate (automatically handles ints):
# f_curve.evaluate(scene.frame_current)
# Interpolation mode:
# [kf.interpolation for kf in f_curve.keyframe_points]
# Property:
# f_curve.data_path
# Property index:
# f_curve.array_index
# Access:
# eval(f_curve.data_path, {"nodes": node_group.nodes})
# Change:
# exec(f_curve.data_path + " = {}".format(f_curve.evaluate(scene.frame_current)), {"nodes": node_group.nodes})
