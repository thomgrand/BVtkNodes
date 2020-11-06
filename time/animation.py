'''
This file is responsible for helping in animating properties until
https://developer.blender.org/T66392
is resolved.
'''
from ..core import *
from ..core import l
import bpy

class AnimationHelper():

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
        if not hasattr(self, 'current_frame') or self.current_frame != current_frame:
            self.current_frame = current_frame
            
        for node_group in bpy.data.node_groups.values():
            node_tree_name = node_group.name
            for key, val in bpy.data.actions.items():

                #Skip unrelated node trees
                if node_tree_name + "Action" in key:
                    self.f_curves = bpy.data.actions["NodeTreeAction"].fcurves
                    self.animated_properties = []
                    self.animated_values = []

                    for f_curve in self.f_curves:
                        new_val = f_curve.evaluate(current_frame)
                        prop_path = f_curve.data_path
                        arr_ind = f_curve.array_index

                        if prop_path not in self.animated_properties:
                            self.animated_properties.append(prop_path)
                            self.animated_values.append(new_val)

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
                        except Exception as ex:
                            l.error("Could not update property " + prop_path + ". Error: " + str(ex))
                    


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