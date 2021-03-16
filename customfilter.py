from .core import l # Import logging
from .core import *
from .time.animation import AnimationHelper

# -----------------------------------------------------------------------------
# Custom filter
# -----------------------------------------------------------------------------


class BVTK_Node_CustomFilter(Node, BVTK_Node):
    '''VTK Custom Filter, defined in Blender text data block. Supports one
    or multiple inputs. Custom function must return a variable which
    is set as input of the node following custom filter.
    '''
    bl_idname = 'BVTK_Node_CustomFilterType'
    bl_label = 'Custom Filter'

    def texts(self, context):
        '''Generate list of text objects to choose'''
        t = []
        i = 0
        for text in bpy.data.texts:
            t.append((text.name, text.name, text.name, 'TEXT', i))
            i += 1
        if not t:
            t.append(('No texts found', 'No texts found', 'No texts found', 'TEXT', i))
        return t

    text: bpy.props.EnumProperty(items=texts, name='text')

    def functions(self, context=None):
        '''Generate list of functions to choose'''
        f = []
        if self.text in bpy.data.texts:
            t = bpy.data.texts[self.text].as_string()
            for func in t.split('def ')[1:]:
                if '(' in func:
                    name = func.split('(')[0].replace(' ','')
                    f.append((name, name, name))
        return f

    func: bpy.props.EnumProperty(items=functions, name='function')

    def m_properties(self):
        return []

    def m_connections(self):
        return (['input'], [], [], ['output'])

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, 'text')
        op = row.operator('node.bvtk_new_text', icon='ZOOM_IN', text='')
        op.name = 'customfilter.py'
        op.body = self.__doc__.replace("    ","")
        if len(self.functions()):
            layout.prop(self, 'func')
        else:
            layout.label(text='No functions found in specified text')

    def apply_properties(self, vtkobj):
        pass

    def apply_inputs(self, vtkobj):
        pass

    def get_output(self, socketname):
        '''Execute user defined function. If something goes wrong,
        print the error and return the input object.
        '''
        input_objects = [x[1] for x in self.get_input_nodes('input')]
        if len(input_objects) == 1:
            input_objects = input_objects[0]
        if self.text in bpy.data.texts:
            t = bpy.data.texts[self.text].as_string()
            try:
                exec(t, globals(), locals())
            except Exception as e:
                l.error('error while parsing user defined text: ' + \
                      str(e).replace('<string>', self.text))
                return self.get_input_node('input')[1]
            if self.func not in locals():
                l.error('function not found')
            else:
                try:
                    user_output = eval(self.func+'(input_objects)')
                    return user_output
                except Exception as e:
                    l.error('error while executing user defined function:' + str(e))
        return self.get_input_node('input')[1]

    def setup(self):
        self.inputs['input'].link_limit = 300

    def export_properties(self):
        '''Export node properties'''
        dict = {}
        if self.text in bpy.data.texts:
            t = bpy.data.texts[self.text].as_string()
            dict['text_as_string'] = t
            dict['text_name'] = self.text
        return dict

    def import_properties(self, dict):
        '''Import node properties'''
        bpy.ops.node.bvtk_new_text(body=dict['text_as_string'], name=dict['text_name'])


class BVTK_OT_NewText(bpy.types.Operator):
    '''New text operator'''
    bl_idname = 'node.bvtk_new_text'
    bl_label = 'Create a new text'

    name: bpy.props.StringProperty(default='New text')
    body: bpy.props.StringProperty()

    def execute(self, context):
        text = bpy.data.texts.new(self.name)
        text.from_string(self.body)
        flag = True
        areas = context.screen.areas
        for area in areas:
            if area.type == 'TEXT_EDITOR':
                for space in area.spaces:
                    if space.type == 'TEXT_EDITOR':
                        if flag:
                            space.text = text
                            space.top = 0
                            flag = False
        if flag:
            self.report({'INFO'}, "See '" + text.name + "' in the text editor")
        return {'FINISHED'}


# ----------------------------------------------------------------
# MultiBlockLeaf
# ----------------------------------------------------------------

class BVTK_Node_MultiBlockLeaf(Node, BVTK_Node):
    '''This node breaks down vtkMultiBlock data and outputs one
    user selected block.
    '''
    bl_idname = 'BVTK_Node_MultiBlockLeafType'
    bl_label = 'Multi Block Leaf'

    def blocks(self, context):
        '''Returns a list for a dynamic enum. Once verified that
        the input vtk object is decomposable in blocks, the list
        will contain an element for every block, with the following
        information:
        - Block index
        - Block data type (ex. structured grid)
        - Block custom name (if it's defined, in most cases it's not)
        '''
        in_node, vtkobj = self.get_input_node('input')
        if not in_node:
            return []
        elif not vtkobj:
            return []
        else:
            vtkobj = resolve_algorithm_output(vtkobj)
            if not vtkobj:
                return []
            if not hasattr(vtkobj, "GetNumberOfBlocks") or not hasattr(vtkobj, "GetBlock"):
                return []
            items = []
            meta_flag = True if hasattr(vtkobj, "GetMetaData") else False
            for i in range(vtkobj.GetNumberOfBlocks()):
                block = vtkobj.GetBlock(i)
                meta_data = vtkobj.GetMetaData(i) if meta_flag else None
                if meta_data:
                    custom_name = meta_data.Get(vtk.vtkCompositeDataSet.NAME())
                    if not custom_name:
                        custom_name = ""
                else:
                    custom_name = ""
                name = "[" + str(i) + "]: " + custom_name + " (" + \
                       (block.__class__.__name__ if block else "Empty Block") + ")"
                items.append((str(i), name, ""))
            return items

    block: bpy.props.EnumProperty(items=blocks, name="Output Block")

    def m_properties(self):
        return []

    def m_connections(self):
        return (['input'], [], [], ['output'])

    def draw_buttons(self, context, layout):
        in_node, vtkobj = self.get_input_node('input')
        if not in_node:
            layout.label(text='Connect a node')
        elif not vtkobj:
            layout.label(text='Input has not vtkobj (try updating)')
        else:
            vtkobj = resolve_algorithm_output(vtkobj)
            if not vtkobj:
                return
            class_name = vtkobj.__class__.__name__
            layout.label(text="Input: "+class_name)
            if not hasattr(vtkobj, "GetNumberOfBlocks") or not hasattr(vtkobj, "GetBlock"):
                layout.label(text="Error: Input Object has no")
                layout.label(text="          MultiBlock Data")
                return
            layout.prop(self, "block")

    def apply_properties(self, vtkobj):
        pass

    def apply_inputs(self, vtkobj):
        pass

    def get_output(self, socketname):
        '''The function checks if the specified block can be retrieved from
        the input vtk object, in case it's possible the said block is returned.
        '''
        in_node, vtkobj = self.get_input_node('input')
        if in_node:
            if vtkobj:
                vtkobj = resolve_algorithm_output(vtkobj)
                if vtkobj:
                    # TODO: remove "not" in front of hasattr(vtkobj, "GetBlock")?
                    if hasattr(vtkobj, "GetNumberOfBlocks") or not hasattr(vtkobj, "GetBlock"):
                        if self.block:
                            return vtkobj.GetBlock(int(self.block))
        return None


# ----------------------------------------------------------------
# Time Selector
# ----------------------------------------------------------------

def update_timestep_in_filename(filename, time_step):
    '''Return file name, where time definition integer string (assumed to
    be located just before dot at end of file name) has been replaced
    to argument time step number
    '''
    import re
    rec1 = re.compile(r'(\d+)\.\w+$', re.M)
    regex1 = rec1.search(filename)
    if regex1:
        numbers = regex1.group(1)
        n = len(numbers)
        defstr = "%0" + str(n) + "d"
        replacement = defstr % time_step
        # Replace with dot at end to increase odds for correct substitution
        newname = filename.replace(numbers + ".", replacement + ".")
        return newname
    return filename


class BVTK_Node_TimeSelector(Node, BVTK_Node):
    '''VTK time management node for time variant data. Display time sets,
    time values and set time.
    '''
    bl_idname = 'BVTK_Node_TimeSelectorType'
    bl_label = 'Time Selector'

    def check_range(self, context):
        in_node, out_port = self.get_input_node('input')
        if in_node:
            if out_port:
                if out_port.IsA('vtkAlgorithmOutput'):
                    prod = out_port.GetProducer()
                    executive = prod.GetExecutive()
                    out_info = prod.GetOutputInformation(out_port.GetIndex())
                    if hasattr(executive, "TIME_STEPS"):
                        time_steps = out_info.Get(executive.TIME_STEPS())

                        # If reader is aware of time, update time step
                        if time_steps:
                            size = len(time_steps)
                            #if self.time_step < -size:
                            #    self.time_step = -size
                            #elif self.time_step >= size:
                            #    self.time_step = size-1
                            # Make data loop outside normal time range.
                            # Value test is needed to avoid infinite property
                            # update loop calling check_range().
                            time_val = self.time_step % size
                            if self.time_step != time_val:
                                self.time_step = time_val

                        # Hack for time unaware readers: If file name of reader
                        # node contains number string at end, update it
                        else:
                            try:
                                filename = in_node.m_FileName
                                newname = update_timestep_in_filename(filename, self.time_step)
                                in_node.m_FileName = newname
                            except Exception as ex:
                                pass

    def activate_scene_time(self, context):
        if self.use_scene_time:
            self.time_step = context.scene.frame_current
            self.check_range(context)


    time_step: bpy.props.IntProperty(update=check_range)
    use_scene_time: bpy.props.BoolProperty(name="Use Scene Time", default=True, update=activate_scene_time)
    b_properties: bpy.props.BoolVectorProperty(name="", size=3, get=BVTK_Node.get_b, set=BVTK_Node.set_b)

    def m_properties(self):
        return ['time_step', 'use_scene_time']

    def m_connections(self):
        return (['input'], [], [], ['output'])

    def draw_buttons(self, context, layout):
        in_node, out_port = self.get_input_node('input')
        if not in_node:
            layout.label(text='Connect a node')
            return
        if not out_port:
            layout.label(text='Input has not vtkobj (try updating)')
            return
        if not out_port.IsA('vtkAlgorithmOutput'):
            layout.label(text='Input is not a vtkAlgorithm')
            return

        prod = out_port.GetProducer()
        executive = prod.GetExecutive()
        out_info = prod.GetOutputInformation(out_port.GetIndex())
        if hasattr(executive, "TIME_STEPS"):
            time_steps = out_info.Get(executive.TIME_STEPS())
            if time_steps:
                row = layout.row()
                row.prop(self, 'time_step', text="Time Step")
                row = layout.row()
                row.prop(self, 'use_scene_time')
                row = layout.row()
                size = len(time_steps)
                row.label(text="Max Steps: "+str(size-1))
                if -size <= self.time_step < size:
                    layout.label(text="Time Value: "+str(time_steps[self.time_step]))
                else:
                    layout.label(text='Index error', icon='ERROR')
            else:
                layout.label(text='No time data on input')
        else:
            layout.label(text='Input contains no time steps')
        return

    def apply_properties(self, vtkobj):
        pass

    def apply_inputs(self, vtkobj):
        pass

    def get_output(self, socketname):
        '''Check if the input is valid and if the time step can be set.
        If tests pass the time step is updated and the input object is returned,
        otherwise None is returned.
        '''
        in_node, out_port = self.get_input_node('input')
        if not in_node or not out_port:
            return None
        if not out_port.IsA('vtkAlgorithmOutput'):
            return None

        prod = out_port.GetProducer()
        executive = prod.GetExecutive()
        out_info = prod.GetOutputInformation(out_port.GetIndex())
        if hasattr(executive, "TIME_STEPS"):
            time_steps = out_info.Get(executive.TIME_STEPS())
            if time_steps:
                size = len(time_steps)
                if -size <= self.time_step < size:
                    if hasattr(prod, "UpdateTimeStep"):
                        prod.UpdateTimeStep(time_steps[self.time_step])
                    else:
                        l.error(prod.__class__.__name__+" does not have 'UpdateTimeStep' method.")
                        l.error("If you can, please document this case and report it to the developers.")
                else:
                    l.error('Index out of time steps range')
        return resolve_algorithm_output(out_port)


# ----------------------------------------------------------------
# Image Data Object Source
# ----------------------------------------------------------------

class BVTK_Node_ImageDataObjectSource(Node, BVTK_Node):
    '''BVTK node to generate a new vtkImageData object'''
    bl_idname = 'BVTK_Node_ImageDataObjectSourceType'
    bl_label = 'VtkImageData Object Source'

    origin: bpy.props.FloatVectorProperty(name='Origin', default=[0.0, 0.0, 0.0], size=3)
    dimensions: bpy.props.IntVectorProperty(name='Dimensions', default=[10, 10, 10], size=3)
    spacing: bpy.props.FloatVectorProperty(name='Spacing', default=[0.1, 0.1, 0.1], size=3)
    multiplier: bpy.props.FloatProperty(name='Multiplier', default=1.0)

    def m_properties(self):
        return ['origin', 'dimensions', 'spacing', 'multiplier']

    def m_connections(self):
        return ([], [], [], ['output'])

    def draw_buttons(self, context, layout):
        layout.prop(self, 'origin')
        layout.prop(self, 'dimensions')
        layout.prop(self, 'spacing')
        layout.prop(self, 'multiplier')

    def apply_properties(self, vtkobj):
        pass

    def get_output(self, socketname):
        '''Generate a new vtkImageData object'''
        from mathutils import Vector
        img = vtk.vtkImageData()
        img.SetOrigin(self.origin)
        c = self.multiplier
        img.SetDimensions([round(c * dim) for dim in self.dimensions])
        img.SetSpacing(Vector(self.spacing) / c)
        return img

# ----------------------------------------------------------------
# Global Time Keeper
# ----------------------------------------------------------------
class BVTK_Node_GlobalTimeKeeper(PersistentStorageUser, AnimationHelper, Node, BVTK_Node):
    '''Global VTK time management node for time variant data. This is used to change
    the speed of the global VTK simulation, updating all Time selectors across the node
    tree according to the currently displayed global time. The VTK time is currently linearly linked
    to the scene time.
    '''
    bl_idname = 'BVTK_Node_GlobalTimeKeeperType'
    bl_label = 'Global Time Keeper'

    def update_time(self, context):
        self.get_persistent_storage()["updated_nodes"] = self.update_animated_properties(context.scene)
        self.get_persistent_storage()["animated_properties"] = self.animated_properties
        self.get_persistent_storage()["interpolation_modes"] = self.interpolation_modes
        self.get_persistent_storage()["animated_values"] = self.animated_values

    def recalculate_steps(self, context):
        pass

    global_time: bpy.props.IntProperty(update=update_time, name="Global Time")
    vtk_speed: bpy.props.FloatProperty(update=recalculate_steps, name="VTK Speed", default=1.0)
    #update_modulo: bpy.props.FloatProperty(update=recalculate_steps)
    #use_scene_time: bpy.props.BoolProperty(name="Use Scene Time", default=True, update=update_time)
    invalid: bpy.props.BoolProperty(name="Is Node Valid")

    #interpolation_items = [(x, x, x) for x in ["BEZIER", "LINEAR"]]
    #interpolation_type: bpy.props.EnumProperty(name="", items=interpolation_items)
    #b_properties: bpy.props.BoolVectorProperty(name="", size=4, get=BVTK_Node.get_b, set=BVTK_Node.set_b)

    #def m_properties(self):
    #    return ['global_time', 'vtk_speed', 'use_scene_time', 'invalid'] #, 'animated_properties']

    def compute_vtk_time(self):
        return self.global_time * self.vtk_speed

    def m_connections(self):
        return ([], [], [], [])

    def draw_buttons(self, context, layout):
        if self.invalid:
            row = layout.row()
            row.label(text="You already have a global time keeper")
            return

        #row = layout.row()
        #row.prop(self, 'use_scene_time')
        row = layout.row()
        #(row.prop(self, 'global_time') if self.use_scene_time else row.label(text="Global Time: {}".format(self.global_time)))
        row.label(text="Global Time: {}".format(self.global_time))
        row = layout.row()
        row.prop(self, 'vtk_speed')
        row = layout.row()
        row.label(text="VTK-Time: {:2f}".format(self.compute_vtk_time()))
        storage = self.get_persistent_storage()
        if "animated_properties" in storage:
            #self.update_animated_properties(bpy.context.scene)
            #storage["animated_properties"] = self.animated_properties
            
            animated_properties = storage["animated_properties"]
            
            if animated_properties is not None and len(animated_properties) > 0:
                row = layout.row()
                row.label(text="Animated properties: ")
                row = layout.row()
                row.label(text="Node")
                row.label(text="Attr.")
                row.label(text="Keyframes")
                row.label(text="Keyframe Vals")
                row.label(text="Current Val")
                row.label(text="Interpol. Mode")
                modes = storage["interpolation_modes"]
                animated_values = storage["animated_values"]

                for elem in animated_properties.values():
                    row = layout.row()
                    [row.label(text=str(val)) for val in elem]
                    #self.interpolation_type = elem[-1]
                    #row.prop(self, 'interpolation_type')
                #for prop, vals, mode in zip(animated_properties, animated_values, modes):
                #    row = layout.row()
                #    row.label(text=prop)
                #    row.label(text=", ".join(["{:.2f}".format(val) for val in vals]))
                #    #row.label(text=mode)

        row = layout.row()
        row.separator()
        row.separator()
        row.separator()
        row.separator()
        row.operator("node.bvtk_node_update", text="update").node_path = node_path(self)
        return

    def apply_properties(self, vtkobj):
        pass

    def apply_inputs(self, vtkobj):
        pass

    def update_cb(self):
        self.update_time(bpy.context)
    
    def set_new_time(self, frame):
        self.global_time = frame
        return self.get_persistent_storage()["updated_nodes"]
        #self.update_animated_properties(bpy.context.scene)

    def get_output(self, socketname):
        '''Check if the input is valid and if the time step can be set.
        If tests pass the time step is updated and the input object is returned,
        otherwise None is returned.
        '''
        in_node, out_port = self.get_input_node('input')
        if not in_node or not out_port:
            return None
        if not out_port.IsA('vtkAlgorithmOutput'):
            return None

        prod = out_port.GetProducer()
        executive = prod.GetExecutive()
        out_info = prod.GetOutputInformation(out_port.GetIndex())
        if hasattr(executive, "TIME_STEPS"):
            time_steps = out_info.Get(executive.TIME_STEPS())
            if time_steps:
                size = len(time_steps)
                if -size <= self.time_step < size:
                    if hasattr(prod, "UpdateTimeStep"):
                        prod.UpdateTimeStep(time_steps[self.time_step])
                    else:
                        l.error(prod.__class__.__name__+" does not have 'UpdateTimeStep' method.")
                        l.error("If you can, please document this case and report it to the developers.")
                else:
                    l.error('Index out of time steps range')
        return resolve_algorithm_output(out_port)

    def setup(self):
        if self.bl_label in persistent_storage["nodes"]:
            self.invalid = True
            raise BVTKException("A Global Time Keeper already exists. There can be only one Global Time Keeper per scene")
        
        AnimationHelper.setup(self)
        assert(self.name == self.bl_label)
        self.bl_label
        persistent_storage["nodes"][self.bl_label] = {} #pass
        self.invalid = False

    def copy(self, node):
        self.setup()

# ----------------------------------------------------------------
# Simplified Transform Filter
# ----------------------------------------------------------------
# TODO
"""
class BVTK_Node_TransformFilterSimple(Node, BVTK_Node):
    bl_idname = 'BVTK_Node_TransformFilterSimpleType'
    bl_label = 'Transform Filter Simple'

    m_Scale: bpy.props.FloatVectorProperty(name='Scale X/Y/Z', default=[1., 1., 1.], size=3)
    m_Rotation: bpy.props.FloatVectorProperty(name='Rotation X/Y/Z', default=[0., 0., 0.],
                                              min=0.,
                                              max=360., size=3)
    m_Translation: bpy.props.FloatVectorProperty(name='Translation X/Y/Z', default=[0., 0., 0.], size=3)

    b_properties: bpy.props.BoolVectorProperty(name="", size=3, get=BVTK_Node.get_b, set=BVTK_Node.set_b)

    def m_properties(self):
        return ['m_Scale', 'm_Rotation', 'm_Translation']

    def m_connections(self):
        return (['input'], ['output'], [], [])

    def get_vtkobj(self):
        return vtk.vtkTransformFilter()

    @run_custom_code
    def apply_properties(self, vtkobj):
        self.vtk_transform = vtk.vtkTransform()
        self.vtk_transform.Scale(*self.m_Scale)
        self.vtk_transform.RotateX(self.m_Rotation[0])
        self.vtk_transform.RotateY(self.m_Rotation[1])
        self.vtk_transform.RotateZ(self.m_Rotation[2])
        self.vtk_transform.Translate(*self.m_Translation)


        vtkobj.SetTransform(self.vtk_transform)
        return 
"""

# Add classes and menu items
TYPENAMES = []
add_class(BVTK_Node_CustomFilter)
TYPENAMES.append('BVTK_Node_CustomFilterType')
add_ui_class(BVTK_OT_NewText)
add_class(BVTK_Node_MultiBlockLeaf)
TYPENAMES.append('BVTK_Node_MultiBlockLeafType')
add_class(BVTK_Node_TimeSelector)
TYPENAMES.append('BVTK_Node_TimeSelectorType')
add_class(BVTK_Node_GlobalTimeKeeper)
TYPENAMES.append('BVTK_Node_GlobalTimeKeeperType')
add_class(BVTK_Node_ImageDataObjectSource)
TYPENAMES.append('BVTK_Node_ImageDataObjectSourceType')
#add_class(BVTK_Node_TransformFilterSimple)
#TYPENAMES.append('BVTK_Node_TransformFilterSimpleType')

menu_items = [NodeItem(x) for x in TYPENAMES]
CATEGORIES.append(BVTK_NodeCategory("Custom", "Custom", items=menu_items))
