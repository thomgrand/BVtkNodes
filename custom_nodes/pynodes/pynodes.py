from ...update import *
from ...core import l # Import logging
from ...core import *
import bmesh
import bpy
import numpy as np
import pyvista as pv
from ...errors.bvtk_errors import BVTKException, assert_bvtk
from ...animation_helper import AnimationHelper, invalid_frame
import vtk
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase
from ...cache import PersistentStorageUser, persistent_storage

try:
    import pyvistaqt as pvqt
    with_pvqt = True
except ImportError as err:
    l.info('PyvistaQT import failed. Preview Node will not be available')
    with_pvqt = False

def backtrack_source_vtkobj(start_node):
    '''Backtracks from the pyvista node to a node that has a valid vtkobj output. 
    This helps to deduce output types so we don't need to update twice in some occassions (e.g. pyvista calculator after CellToPointData)
    '''
    #We just look at the first input
    try:
        if len(start_node.inputs) > 0 and len(start_node.inputs[0].links) > 0:
            link = start_node.inputs[0].links[0]
            source_node = link.from_node
            if source_node is not None:
                vtkobj = source_node.get_output('output')

                if vtkobj == 0 or vtkobj is None:
                    return backtrack_source_vtkobj(source_node)
                else:
                    return vtkobj
    except Exception as ex:
        pass

    return None

class ArbitraryInputsHelper():
    '''Helps in organizing nodes with an arbitrary number of input nodes that will
    occur and disappear according to the connections made. See Pyvista-Calculator Node
    for an example
    '''
    input_limit = 10
    input_format_str = "inputs[{}]"

    @property
    def current_inputs(self):
        return len(self.inputs)

    @property
    def current_active_inputs(self):
        return len(self.connected_input_nodes)

    @property
    def available_input_nodes(self):
        return [self.input_format_str.format(i) for i in range(max(1, self.current_inputs))]

    @property
    def connected_input_nodes(self):
        in_info, in_nodes, vtkobjs = self.get_in_infos()
        if np.sum(in_nodes != 0) < self.input_limit:
            return self.available_input_nodes[:-1] #Last one is dangling
        else:
            return self.available_input_nodes #All nodes connected
        

    def m_connections(self):
        return ( self.available_input_nodes, [], [], ['output'] )

    def update_input_names(self):
        for input_i in range(self.current_inputs):
            self.inputs[input_i].name = self.input_format_str.format(input_i)

    def get_in_infos(self):
        input_names = self.available_input_nodes
        in_info = [self.get_input_node(input_name) for input_name in input_names]
        in_nodes, vtkobjs = zip(*in_info)
        in_nodes = np.array(in_nodes)

        return in_info, in_nodes, vtkobjs

    def update(self):
        current_inputs = self.current_inputs
        input_names = self.available_input_nodes
        
        in_info, in_nodes, vtkobjs = self.get_in_infos()
        
        old_connections_modified = (in_nodes[:-1] == 0) # | (in_nodes[:-1] == None)
        new_connection = in_nodes[-1] not in [0, None]
        if np.any(old_connections_modified):
            remove_pos = np.where(old_connections_modified)[0]

            for del_i, del_name in zip(remove_pos, [input_names[i] for i in remove_pos]):
                self.inputs.remove(self.inputs[del_name])

            self.update_input_names() #To remove gaps in the input numbering
        
        elif new_connection and current_inputs < self.input_limit:
            self.inputs.new('BVTK_NodeSocketType', self.input_format_str.format(current_inputs))


class CustomPyvistaFilter(VTKPythonAlgorithmBase):
    def __init__(self, n_input_ports, output_type, filter_code, time_steps=None):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=n_input_ports,
            nOutputPorts=1, outputType=output_type,
            additional_dict=None)
        self.time_steps = time_steps
        self.filter_code = filter_code
        if additional_dict is None:
            additional_dict = {}
        
        self.additional_dict = additional_dict
        self.output = vtk_pv_mapping[output_type]()


    def RequestInformation(self, request, in_info, out_info):
        if self.time_steps is not None:
            algorithm = self
            #info = out_info.GetInformationObject(0)
            executive = algorithm.GetExecutive()
            out_info = executive.GetOutputInformation(0)
            out_info.Remove(executive.TIME_STEPS())
            for timestep in self.time_steps:
                out_info.Append(executive.TIME_STEPS(), timestep)
            out_info.Remove(executive.TIME_RANGE())
            out_info.Append(executive.TIME_RANGE(), self.time_steps[0])
            out_info.Append(executive.TIME_RANGE(), self.time_steps[-1])

        return 1

    def RequestData(self, request, in_info, out_info):
        def GetUpdateTimestep(algorithm):
            """Returns the requested time value, or None if not present"""
            executive = algorithm.GetExecutive()
            #out_info = executive.GetOutputInformation(0)
            if not out_info.Has(executive.UPDATE_TIME_STEP()):
                return None
            return out_info.Get(executive.UPDATE_TIME_STEP())

        try:
            exec(self.filter_code, locals={**locals(), **self.additional_dict})
        except Exception as ex:
            raise BVTKException("Execution of the custom pyvista filter failed", ex)

        self.output.GetInformation().Set(self.output.DATA_TIME_STEP(), GetUpdateTimestep(self))
        return 1


class BVTK_Node_ArbitraryInputsTest(ArbitraryInputsHelper, Node, BVTK_Node):
    '''Test Node to see the functionality of ArbitraryInputsHelper
    '''
    bl_idname = 'BVTK_Node_ArbitraryInputsTestType' # type name
    bl_label  = 'Arbitrary Inputs Test' # label for nice name display

    input_limit = 3
    def draw_buttons(self, context, layout):
        row = layout.row()
        row.label(text = ", ".join(self.available_input_nodes))

    def m_properties(self):
        return []

    def setup(self):
        print("Setup called")
        pass

    def apply_inputs(self, vtkobj):
        print("Apply called")
        pass
        """
        added = [vtkobj.GetInputConnection(0, i) for i in range(vtkobj.GetNumberOfInputConnections(0))]
        toadd = []
        for node, in_obj in self.get_input_nodes('input'):
            toadd.append(in_obj)
            if in_obj not in added:
                vtkobj.AddInputConnection(in_obj)

        for obj in added:
            if obj not in toadd:
                vtkobj.RemoveInputConnection(0, obj)
        """

vtk_pv_mapping = (
    {
        'vtkUnstructuredGrid': pv.UnstructuredGrid,
        'vtkPolyData': pv.PolyData,
        'vtkImageData': pv.UniformGrid,
        'vtkRectilinearGrid': pv.RectilinearGrid,
        'vtkStructuredGrid': pv.StructuredGrid
    })

def map_vtk_to_pv_obj(vtkobj):
    if not is_mappable_pyvista_type(vtkobj):
        return None
    else:
        return vtk_pv_mapping[vtkobj.GetClassName()]

def is_mappable_pyvista_type(vtkobj):
    class_name = vtkobj.GetClassName()
    return class_name in vtk_pv_mapping

def is_pyvista_obj(vtkobj):
    vtk_pv_mask, vtk_types, pv_types = np.array([(isinstance(vtkobj, pv_type), vtk, pv_type) for vtk_type, pv_type in vtk_pv_mapping.items()]).T
    return ((vtk_types[np.where(vtk_pv_mask)][0], pv_types[np.where(vtk_pv_mask)][0]) if np.any(vtk_pv_mask) else None)

def create_pyvista_wrapper(vtkobj):
    if isinstance(vtkobj, vtk.vtkAlgorithmOutput):
        vtkobj = resolve_algorithm_output(vtkobj)

    try:
        pvobj = vtk_pv_mapping[vtkobj.GetClassName()]
    except KeyError as err:
        #Try reflection based mapping
        try:
            pvobj = getattr(pv, vtkobj.GetClassName()[3:])
        except Exception as ex:
            raise BVTKException("Conversion VTK -> Pyvista failed", ex)

    return pvobj(vtkobj)

def convert_vtk_to_pv(vtkobj):
    """Converts the VTK object to the corresponding pyvista type

    Parameters
    ----------
    vtkobj : pyvista-obj or vtk-obj
        Object to be converted to a pyvista object if not already converted

    Returns
    -------
    pyvista-obj
        Pyvista object of converted vtk type

    Raises
    ------
    BVTKException
        If non supported types are encountered
    """
    if is_pyvista_obj(vtkobj):
        pvobj = vtkobj
    elif is_mappable_pyvista_type(vtkobj):
        pvobj = map_vtk_to_pv_obj(vtkobj)(vtkobj)
    else:
        raise BVTKException("VTK Object of type %s not supported. Current support is limited to: %s" % (type(vtkobj), vtk_pv_mapping.values()))

    return pvobj


class BVTK_Node_PyvistaSource(Node, BVTK_Node):
    '''Create meshes using the numpy and pyvista interfaces. The output type
    can be specified for easier organization.
    '''
    bl_idname = 'BVTK_Node_PyvistaSourceType' # type name
    bl_label  = 'Pyvista Source' # label for nice name display
    """
    m_Name: bpy.props.StringProperty(name='Name', default='mesh')
    create_all_verts: bpy.props.BoolProperty(name='Create All Verts', default=False)
    create_edges: bpy.props.BoolProperty(name='Create Edges', default=True)
    create_faces: bpy.props.BoolProperty(name='Create Faces', default=True)
    smooth: bpy.props.BoolProperty(name='Smooth', default=False)
    recalc_norms: bpy.props.BoolProperty(name='Recalculate Normals', default=False)
    generate_material: bpy.props.BoolProperty(name='Generate Material', default=False)

    def start_scan(self, context):
        if context:
            if self.auto_update:
                bpy.ops.node.bvtk_auto_update_scan(
                    node_name=self.name,
                    tree_name=context.space_data.node_tree.name)

    auto_update: bpy.props.BoolProperty(default=False, update=start_scan)

    def m_properties(self):
        return ['m_Name', 'create_all_verts', 'create_edges', 'create_faces',
                'smooth', 'recalc_norms', 'generate_material']

    def m_connections(self):
        return ( ['input'],[],[],[] )

    def draw_buttons(self, context, layout):
        layout.prop(self, 'm_Name')
        layout.prop(self, 'create_all_verts')
        layout.prop(self, 'create_edges')
        layout.prop(self, 'create_faces')
        layout.prop(self, 'auto_update', text='Auto update')
        layout.prop(self, 'smooth', text='Smooth')
        layout.prop(self, 'recalc_norms')
        layout.prop(self, 'generate_material')
        layout.separator()
        layout.operator("node.bvtk_node_update", text="update").node_path = node_path(self)

    def update_cb(self):
        '''Update node color bar and update Blender object'''
        input_node, vtkobj = self.get_input_node('input')
        ramp = None
        if input_node and input_node.bl_idname == 'BVTK_Node_ColorMapperType':
            ramp = input_node
            ramp.update()    # setting auto range
            input_node, vtkobj = input_node.get_input_node('input')
        if vtkobj:
            vtkobj = resolve_algorithm_output(vtkobj)
            vtkdata_to_blender_mesh (vtkobj, self.m_Name, smooth=self.smooth,
                                     create_all_verts=self.create_all_verts,
                                     create_edges=self.create_edges,
                                     create_faces=self.create_faces,
                                     recalc_norms=self.recalc_norms,
                                     generate_material=self.generate_material,
                                     ramp=ramp)
            update_3d_view()

    def apply_properties(self, vtkobj):
        pass
    """
    pass

class BVTK_Node_PyvistaFilter(ArbitraryInputsHelper, PersistentStorageUser, Node, BVTK_Node):
    bl_idname = 'BVTK_Node_PyvistaFilterType' # type name
    bl_label  = 'Pyvista Filter/Source' # label for nice name display

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

    filter_code: bpy.props.EnumProperty(items=texts, name='Filter Code')
    output_type_items = [(x, x, x) for x in vtk_pv_mapping.keys()]
    output_type: bpy.props.EnumProperty(name='Output Type', default=list(vtk_pv_mapping.keys())[0], 
                                    items=output_type_items)
    add_time: bpy.props.BoolProperty(name='Add Time', default=False)
    time_code: bpy.props.EnumProperty(items=texts, name='Time Code')

    #current_output = None
    def m_properties(self):
        return ['filter_code', 'output_type', 'add_time', 'time_code']

    b_properties: bpy.props.BoolVectorProperty(name="", size=4, get=BVTK_Node.get_b, set=BVTK_Node.set_b)
    #info_msg = ""

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

    def create_input_dicts(self, pvobjs):
        input_dicts = []
        for pvobj, input_i in zip(pvobjs, range(self.current_active_inputs)):
            single_input_dict = pvobj
            for data_arr_name in ["point", "cell", "field"]:
                array_name = data_arr_name + "_arrays"
                key_name = data_arr_name.capitalize() + "Data"
                data_dict = {k: v for k, v in getattr(pvobj, array_name).items()}
                setattr(single_input_dict, key_name, data_dict)

            input_dicts.append(single_input_dict)

        return input_dicts

    def apply_properties(self, vtkobj):
        print("Apply properties called with " + str(vtkobj))
        assert_bvtk(type(vtkobj) in vtk_pv_mapping.values(), "Expected a pyvista type. This is an internal error... try updating")

        (data_dicts, pvobjs, vtkobjs) = self.get_data_dicts()
        vtkobj.deep_copy(pvobjs[0]) #Take shape and arguments from the input

        lambda_eval_str = "lambda: " + self.m_LambdaCode

        local_dict = {"inputs": self.create_input_dicts(pvobjs)}
        local_dict["numpy"] = np
        local_dict["np"] = np

        if AnimationHelper.current_frame != invalid_frame:
            local_dict["frame"] = AnimationHelper.current_frame

        #if AnimationHelper.vtk_time != invalid_vtk_time:
        #    local_dict["time"] = AnimationHelper.vtk_time
        #local_dict["min_time"] = min_time
        #local_dict["max_time"] = max_time

        #Only one input: We can unpack the chosen dictionary
        if len(data_dicts) == 1:
            local_dict = {**local_dict, **{k: v for k, v in data_dicts[0].items()}}

        try:
            result = eval(lambda_eval_str, locals=local_dict)()
        except Exception as ex:
            err_msg = "Execution of the given calculator function failed with: {}".format(ex)
            l.error(err_msg)
            raise BVTKException(err_msg, ex)


        if self.e_AttrType == 'PointData':
            target_dict = vtkobj.point_arrays
            expected_size = vtkobj.n_points
        elif self.e_AttrType == 'CellData':
            target_dict = vtkobj.cell_arrays
            expected_size = vtkobj.n_cells
        else:
            target_dict = vtkobj.field_arrays
            expected_size = None

        assert_bvtk(isinstance(result, np.ndarray) and (expected_size is None or result.shape[0] == expected_size), 
                        "Expected a [%d (, x)] array as a result of the lambda expression. Got %s" % (expected_size, result))
        target_dict[self.m_ResultName] = result #Change the mutable dictionary in-place
        persistent_storage["nodes"][self.name] = vtkobj

        pass

    def apply_inputs(self, vtkobj):
        print("Apply input called with " + str(vtkobj))
        pass

    def input_is_invalid(self, vtkobjs):
        return len(vtkobjs) == 0 or vtkobjs[0] is None or vtkobjs[0] == 0

    def get_vtkobj(self):
        vtkobjs = self.get_input_vtkobjs()
        if self.input_is_invalid(vtkobjs):
            return None
        
        ret_tuples = self.get_data_dicts(vtkobjs)
        if ret_tuples is not None:
            (data_dicts, pvobjs, vtkobjs) = ret_tuples
            return pvobjs[0]
        else:
            return None

    def get_output(self, socketname):
        '''Execute user defined function. If something goes wrong,
        print the error and return the input object.
        '''
        #if len(self.m_LambdaCode) == 0:
        #    return None
        if self.name in persistent_storage["nodes"]:
            return persistent_storage["nodes"][self.name]

        vtkobjs = self.get_input_vtkobjs()
        if self.input_is_invalid(vtkobjs):
            return None

        #Output is equivalent to the type of the first input
        vtkobj = vtkobjs[0]
        pvobj = convert_vtk_to_pv(vtkobj)
        #(data_dicts, pvobjs, vtkobjs) = self.get_data_dicts(vtkobjs)
        return pvobj
        #return vtkobjs[0]

    def get_input_vtkobjs(self):
        nr_inputs = self.current_active_inputs
        vtkobjs = []

        for input_i in range(nr_inputs):
            in_node, vtkobj = self.get_input_node(self.inputs[input_i].name)
            if vtkobj:
                vtkobjs.append(resolve_algorithm_output(vtkobj))

        return vtkobjs

    def get_data_dicts(self, vtkobjs=None):
        nr_inputs = self.current_active_inputs

        data_dicts = []
        pvobjs = []

        if vtkobjs is None:
            vtkobjs = self.get_input_vtkobjs()

        for input_i in range(nr_inputs):
            #in_node, vtkobj = self.get_input_node(self.inputs[input_i].name)
            #vtkobjs.append(resolve_algorithm_output(vtkobj))

            #At least one of the objects is not resolvable currently
            if vtkobjs[input_i] is None or vtkobjs[input_i] == 0:
                return None

            pvobj = create_pyvista_wrapper(vtkobjs[input_i])

            if self.e_AttrType == 'PointData':
                data_dict = pvobj.point_arrays
            elif self.e_AttrType == 'CellData':
                data_dict = pvobj.cell_arrays
            else:
                data_dict = pvobj.field_arrays

            data_dicts.append(data_dict)
            pvobjs.append(pvobj)

        return (data_dicts, pvobjs, vtkobjs)

class BVTK_Node_PyvistaCalculator(ArbitraryInputsHelper, PersistentStorageUser, Node, BVTK_Node):
    bl_idname = 'BVTK_Node_PyvistaCalculatorType' # type name
    bl_label  = 'Pyvista Calculator' # label for nice name display

    m_LambdaCode:   bpy.props.StringProperty (name='Execution Code', default='')
    m_ResultName:   bpy.props.StringProperty (name='Result Name', default='result')

    e_AttrType_items = [ (x,x,x) for x in ['PointData', 'CellData', 'FieldData']]
    e_AttrType:   bpy.props.EnumProperty   (name='Attribute Type', default='PointData', items=e_AttrType_items)
    #m_InfoMsg: bpy.props.StringProperty (name='Info Message', default='') #, is_readonly=True)

    #current_output = None
    def m_properties(self):
        return ['m_LambdaCode', 'm_ResultName', 'e_AttrType']

    b_properties: bpy.props.BoolVectorProperty(name="", size=3, get=BVTK_Node.get_b, set=BVTK_Node.set_b)
    #info_msg = ""

    #def draw_buttons(self, context, layout):
        #if len(self.m_LambdaCode) == 0:
    #    row = layout.row()
    #    row.label(text = self.info_msg)

    def create_input_dicts(self, pvobjs):
        input_dicts = []
        for pvobj, input_i in zip(pvobjs, range(self.current_active_inputs)):
            single_input_dict = pvobj
            for data_arr_name in ["point", "cell", "field"]:
                array_name = data_arr_name + "_arrays"
                key_name = data_arr_name.capitalize() + "Data"
                data_dict = {k: v for k, v in getattr(pvobj, array_name).items()}
                setattr(single_input_dict, key_name, data_dict)

            input_dicts.append(single_input_dict)

        return input_dicts

    def apply_properties(self, vtkobj):
        print("Apply properties called with " + str(vtkobj))
        assert_bvtk(type(vtkobj) in vtk_pv_mapping.values(), "Expected a pyvista type. This is an internal error... try updating")

        (data_dicts, pvobjs, vtkobjs) = self.get_data_dicts()
        vtkobj.deep_copy(pvobjs[0]) #Take shape and arguments from the input

        lambda_eval_str = "lambda: " + self.m_LambdaCode

        local_dict = {"inputs": self.create_input_dicts(pvobjs)}
        local_dict["numpy"] = np
        local_dict["np"] = np

        if AnimationHelper.current_frame != invalid_frame:
            local_dict["frame"] = AnimationHelper.current_frame

        #if AnimationHelper.vtk_time != invalid_vtk_time:
        #    local_dict["time"] = AnimationHelper.vtk_time
        #local_dict["min_time"] = min_time
        #local_dict["max_time"] = max_time

        #Only one input: We can unpack the chosen dictionary
        if len(data_dicts) == 1:
            local_dict = {**local_dict, **{k: v for k, v in data_dicts[0].items()}}

        try:
            result = eval(lambda_eval_str, local_dict)()
        except Exception as ex:
            err_msg = "Execution of the given calculator function failed with: {}, {}".format(ex.__class__.__name__, ex)
            l.error(err_msg)
            raise BVTKException(err_msg, ex)


        if self.e_AttrType == 'PointData':
            target_dict = vtkobj.point_arrays
            expected_size = vtkobj.n_points
        elif self.e_AttrType == 'CellData':
            target_dict = vtkobj.cell_arrays
            expected_size = vtkobj.n_cells
        else:
            target_dict = vtkobj.field_arrays
            expected_size = None

        assert_bvtk(isinstance(result, np.ndarray) and (expected_size is None or result.shape[0] == expected_size), 
                        "Expected a [%d (, x)] array as a result of the lambda expression. Got %s" % (expected_size, result))
        target_dict[self.m_ResultName] = result #Change the mutable dictionary in-place
        persistent_storage["nodes"][self.name] = vtkobj

        pass

    def apply_inputs(self, vtkobj):
        print("Apply input called with " + str(vtkobj))
        pass

    def input_is_invalid(self, vtkobjs):
        return len(vtkobjs) == 0 or vtkobjs[0] is None or vtkobjs[0] == 0

    def get_vtkobj(self):
        vtkobjs = self.get_input_vtkobjs()
        if self.input_is_invalid(vtkobjs):
            return None
        
        ret_tuples = self.get_data_dicts(vtkobjs)
        if ret_tuples is not None:
            (data_dicts, pvobjs, vtkobjs) = ret_tuples
            return pvobjs[0]
        else:
            return None

    def get_output(self, socketname):
        '''Execute user defined function. If something goes wrong,
        print the error and return the input object.
        '''
        #if len(self.m_LambdaCode) == 0:
        #    return None
        if self.name in persistent_storage["nodes"]:
            return persistent_storage["nodes"][self.name]

        vtkobjs = self.get_input_vtkobjs()
        if self.input_is_invalid(vtkobjs):
            return None

        #Output is equivalent to the type of the first input
        vtkobj = vtkobjs[0]
        pvobj = convert_vtk_to_pv(vtkobj)
        #(data_dicts, pvobjs, vtkobjs) = self.get_data_dicts(vtkobjs)
        return pvobj
        #return vtkobjs[0]

    def get_input_vtkobjs(self):
        nr_inputs = self.current_active_inputs
        vtkobjs = []

        for input_i in range(nr_inputs):
            in_node, vtkobj = self.get_input_node(self.inputs[input_i].name)
            if vtkobj:
                vtkobjs.append(resolve_algorithm_output(vtkobj))

        return vtkobjs

    def get_data_dicts(self, vtkobjs=None):
        nr_inputs = self.current_active_inputs

        data_dicts = []
        pvobjs = []

        if vtkobjs is None:
            vtkobjs = self.get_input_vtkobjs()

        for input_i in range(nr_inputs):
            #in_node, vtkobj = self.get_input_node(self.inputs[input_i].name)
            #vtkobjs.append(resolve_algorithm_output(vtkobj))

            #At least one of the objects is not resolvable currently
            if vtkobjs[input_i] is None or vtkobjs[input_i] == 0:
                return None

            pvobj = create_pyvista_wrapper(vtkobjs[input_i])

            if self.e_AttrType == 'PointData':
                data_dict = pvobj.point_arrays
            elif self.e_AttrType == 'CellData':
                data_dict = pvobj.cell_arrays
            else:
                data_dict = pvobj.field_arrays

            data_dicts.append(data_dict)
            pvobjs.append(pvobj)

        return (data_dicts, pvobjs, vtkobjs)

    #def setup(self):
    #    print("Setup called")
    #    pass

class BVTK_Node_SetActiveArrays(PersistentStorageUser, Node, BVTK_Node):
    '''Convenience node that helps in setting the active scalars, vectors or tensors
    '''
    bl_idname = 'BVTK_Node_SetActiveArraysType'
    bl_label  = 'Set Active Arrays'

    e_attr_items = [ (x,x,x) for x in ['Point', 'Cell']]
    e_attr_type:   bpy.props.EnumProperty   (name='Attribute Type', default='Point', items=e_attr_items)

    e_component_items = [ (x,x,x) for x in ['Scalars', 'Vectors', 'Tensors']]
    e_component_type:   bpy.props.EnumProperty   (name='Attribute Type', default='Scalars', items=e_component_items)

    m_array_name:   bpy.props.StringProperty (name='Array Name', default='')
    b_deep_copy: bpy.props.BoolProperty(name='Deep Copy', default=True)

    b_properties: bpy.props.BoolVectorProperty(name="", size=5, get=BVTK_Node.get_b, set=BVTK_Node.set_b)

    def m_properties(self):
        return ['e_attr_type', 'e_component_type', 'm_array_name', 'b_deep_copy']

    def m_connections(self):
        return (['input'],[],[],['output'])

    def apply_properties(self, vtkobj_ignored):
        in_node, vtkobj = self.get_input_node('input')
        vtkobj = resolve_algorithm_output(vtkobj)
        pvobj = convert_vtk_to_pv(vtkobj)

        if self.b_deep_copy:
            pvobj_copied = type(pvobj)()
            pvobj_copied.deep_copy(pvobj)
            pvobj = pvobj_copied

        selected_arrays = getattr(pvobj, self.e_attr_type.lower() + "_arrays")
        assert_bvtk(self.m_array_name in selected_arrays, "Requested array of name %s not found in %s" % (self.m_array_name, selected_arrays.keys()))

        try:
            active_attr = getattr(pvobj, "set_active_" + self.e_component_type.lower())(self.m_array_name)
        except Exception as ex:
            raise BVTKException("Could not set the active " + self.e_component_type, ex)
        
        persistent_storage["nodes"][self.name] = pvobj

    def apply_inputs(self, vtkobj):
        print("Apply input called with " + str(vtkobj))
        pass

    def get_vtkobj(self):
        return self.get_output("output")

    def get_output(self, socketname):
        if self.name in persistent_storage["nodes"]:
            return persistent_storage["nodes"][self.name]

        in_node, vtkobj = self.get_input_node('input')
        if vtkobj != 0 and vtkobj is not None:
            return vtkobj
        #Some nodes do not provide the correct output type immediately. Yet if we return None,
        #the node will not be executed.
        elif in_node != 0 and in_node is not None: 
            return backtrack_source_vtkobj(in_node)
        else:
            return None

#def 

class BVTK_Node_Preview(PersistentStorageUser, Node, BVTK_Node):
    '''BVTK Preview Node'''
    bl_idname = 'BVTK_Node_PreviewType'
    bl_label  = 'Preview'

    def m_properties(self):
        return []

    def m_connections(self):
        return (['input'],[],[],['output'])

    def update_cb(self):
        in_node, vtkobj = self.get_input_node('input')
        class_name = vtkobj.__class__.__name__
        pvobj = None

        #Need to be converted
        if class_name in vtk_pv_mapping:
            pvobj = vtk_pv_mapping[class_name](vtkobj)

        #Already converted
        elif np.any([isinstance(vtkobj, val) for val in vtk_pv_mapping.values()]):
            pvobj = vtkobj

        storage = self.get_persistent_storage()
        if "background_plotter" not in storage:
            storage["background_plotter"] = pvqt.BackgroundPlotter()
        
        plotter = storage["background_plotter"]

        #TODO: Still crashes
        if pvobj is not None:
            #pvobj.plot()
            plotter.add_mesh(pvobj)
            plotter.show_bounds(grid=True, location='back')

        #pvobj

    def draw_buttons(self, context, layout):
        fs="{:.5g}" # Format string
        in_node, vtkobj = self.get_input_node('input')
        if not in_node:
            layout.label(text='Connect a node')
        elif not vtkobj:
            layout.label(text='Input has not vtkobj (try updating)')
        elif vtkobj.__class__.__name__ not in vtk_pv_mapping:
            layout.label(text='Unsupported vtk type (%s)' % (vtkobj.__class__.__name__))
        else:
            vtkobj = resolve_algorithm_output(vtkobj)
            if not vtkobj:
                return

            layout.label(text='Type: ' + vtkobj.__class__.__name__)

        layout.separator()
        row = layout.row()
        row.separator()
        row.separator()
        row.separator()
        row.separator()
        row.operator("node.bvtk_node_update", text="preview").node_path = node_path(self)

    def apply_properties(self, vtkobj):
        pass

    def get_output(self, socketname):
        return self.get_input_node('input')[0]

class BVTK_Node_BlenderToVTK(Node, BVTK_Node):
    '''BVTK BlenderToVTK Node'''
    bl_idname = 'BVTK_Node_BlenderToVTKType'
    bl_label  = 'BlenderToVTK'

    input_mesh_prop: bpy.props.PointerProperty(type=bpy.types.Object)
    #test_prop: bpy.props.PointerProperty(type=bpy.types.PropertyGroup)
    output_type_items = [ (x,x,x) for x in ['UnstructuredGrid', 'PolyData']]
    output_type_prop:   bpy.props.EnumProperty   (name='Output Type', default='UnstructuredGrid', items=output_type_items)
    b_properties: bpy.props.BoolVectorProperty(name="", size=2, get=BVTK_Node.get_b, set=BVTK_Node.set_b)

    def m_properties(self):
        return ['input_mesh_prop', 'output_type_prop']

    def m_connections(self):
        return ([],[],[],['output'])

    def update_cb(self):
        in_node, vtkobj = self.get_input_node('input')
        class_name = vtkobj.__class__.__name__
        pvobj = None

        #Need to be converted
        if class_name in vtk_pv_mapping:
            pvobj = vtk_pv_mapping[class_name](vtkobj)

        #Already converted
        elif np.any([isinstance(vtkobj, val) for val in vtk_pv_mapping.values()]):
            pvobj = vtkobj

        #TODO: Segfault on windows
        #if pvobj is not None:
        #    pvobj.plot()

        pvobj

    """
    def draw_buttons(self, context, layout):
        fs="{:.5g}" # Format string
        in_node, vtkobj = self.get_input_node('input')
        if not in_node:
            layout.label(text='Connect a node')
        elif not vtkobj:
            layout.label(text='Input has not vtkobj (try updating)')
        elif vtkobj.__class__.__name__ not in vtk_pv_mapping:
            layout.label(text='Unsupported vtk type (%s)' % (vtkobj.__class__.__name__))
        else:
            vtkobj = resolve_algorithm_output(vtkobj)
            if not vtkobj:
                return

            layout.label(text='Type: ' + vtkobj.__class__.__name__)

        layout.separator()
        row = layout.row()
        row.separator()
        row.separator()
        row.separator()
        row.separator()
        row.operator("node.bvtk_node_update", text="preview").node_path = node_path(self)
    """

    def apply_properties(self, vtkobj):
        pass

    def get_output(self, socketname):
        return None

"""
tmp = [(len(mesh.polygons[i].vertices[:]), mesh.polygons[i].vertices[:]) for i in range(len(mesh.polygons))]
lens, faces = zip(*tmp)
np.array(lens) == 3 #A
nr_vertices = len(mesh.vertices)
points = np.zeros([nr_vertices, 3], dtype=np.float32)
mesh.vertices.foreach_get('co', points.ravel())
"""

#def register_nodes():
l.info("Registering Pyvista nodes")
# Add classes and menu items
TYPENAMES = []
add_class(BVTK_Node_PyvistaSource)
TYPENAMES.append('BVTK_Node_PyvistaSourceType')
add_class(BVTK_Node_ArbitraryInputsTest)
TYPENAMES.append('BVTK_Node_ArbitraryInputsTestType')
add_class(BVTK_Node_PyvistaCalculator)
TYPENAMES.append('BVTK_Node_PyvistaCalculatorType')
add_class(BVTK_Node_SetActiveArrays)
TYPENAMES.append('BVTK_Node_SetActiveArraysType')

if with_pvqt:
    add_class(BVTK_Node_Preview)
    TYPENAMES.append('BVTK_Node_PreviewType')

add_class(BVTK_Node_BlenderToVTK)
TYPENAMES.append('BVTK_Node_BlenderToVTKType')
"""
add_class(BVTK_Node_PyvistaFilter)
TYPENAMES.append('BVTK_Node_PyvistaFilter')
add_class(BVTK_Node_PyvistaCalculator)
TYPENAMES.append('BVTK_Node_PyvistaCalculator')
"""
menu_items = [NodeItem(x) for x in TYPENAMES]
CATEGORIES.append(BVTK_NodeCategory("Pyvista", "Pyvista", items=menu_items))
