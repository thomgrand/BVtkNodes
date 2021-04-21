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


class PyvistaComputeHelper(ArbitraryInputsHelper, PersistentStorageUser):

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

    def apply_inputs(self, vtkobj):
        print("Apply input called with " + str(vtkobj))
        pass

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

        data_dicts = {"point_arrays": [], "cell_arrays": [], "field_arrays": []}
        pvobjs = []

        if vtkobjs is None:
            vtkobjs = self.get_input_vtkobjs()

        for input_i in range(nr_inputs):
            #At least one of the objects is not resolvable currently
            if vtkobjs[input_i] is None or vtkobjs[input_i] == 0:
                return None

            pvobj = create_pyvista_wrapper(vtkobjs[input_i])

            data_dicts["point_arrays"].append(pvobj.point_arrays)
            data_dicts["cell_arrays"].append(pvobj.cell_arrays)
            data_dicts["field_arrays"].append(pvobj.field_arrays)
            pvobjs.append(pvobj)

        return (data_dicts, pvobjs, vtkobjs)

    #def setup(self):
    #    print("Setup called")
    #    pass

class BVTK_Node_PyvistaSource(PyvistaComputeHelper, Node, BVTK_Node):
    '''Create meshes using the numpy and pyvista interfaces. Can take multiple inputs similar to PyvistaCalculator
    '''
    bl_idname = 'BVTK_Node_PyvistaSourceType' # type name
    bl_label  = 'Pyvista Source' # label for nice name display

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

    script: bpy.props.EnumProperty(items=texts, name='Script')    
    #output_type_items = [ (x,x,x) for x in ['UnstructuredGrid', 'PolyData']]
    #output_type_prop:   bpy.props.EnumProperty   (name='Output Type', default='UnstructuredGrid', items=output_type_items)

    def m_properties(self):
        return ['script'] #, 'output_type_prop']

    b_properties: bpy.props.BoolVectorProperty(name="", size=2, get=BVTK_Node.get_b, set=BVTK_Node.set_b)

    def apply_properties(self, vtkobj):
        print("Apply properties called with " + str(vtkobj))
        assert_bvtk(self.script in bpy.data.texts, "Please provide a script for Pyvista Source")

        script_str = bpy.data.texts[self.script].as_string()
        script_str_indented = "\n".join(['\t' + line for line in script_str.splitlines()])
        (data_dicts, pvobjs, vtkobjs) = self.get_data_dicts()
        func_header_str = "def executePyvistaSource(inputs, frame):\n"
        func_str = func_header_str + script_str_indented

        #local_dict = {"inputs": self.create_input_dicts(pvobjs)}
        inputs = self.create_input_dicts(pvobjs)
        #local_dict["numpy"] = np
        #local_dict["np"] = np
        frame = AnimationHelper.current_frame

        #if AnimationHelper.current_frame != invalid_frame:
        #    local_dict["frame"] = AnimationHelper.current_frame

        #if AnimationHelper.vtk_time != invalid_vtk_time:
        #    local_dict["time"] = AnimationHelper.vtk_time
        #local_dict["min_time"] = min_time
        #local_dict["max_time"] = max_time

        #Only one input: We can unpack the chosen dictionary
        #if len(data_dicts) == 1:
        #    local_dict = {**local_dict, **{k: v for k, v in data_dicts[0].items()}}

        try:
            #Execute the function to create it in a temporary scope
            scope = {**globals(), **locals()} #Make local and global variables visible to the function (e.g. numpy and pyvista)
            exec(func_str, scope)
            result = scope['executePyvistaSource'](inputs, frame)
        except Exception as ex:
            err_msg = "Execution of the given Pyvista Source function failed with: {}, {}".format(ex.__class__.__name__, ex)
            l.error(err_msg)
            raise BVTKException(err_msg, ex)

        #TODO: Better possibility to check if the return value is a vtk object? Not all vtk objects may have GetPoints
        assert_bvtk(result is not None and hasattr(result, 'GetPoints'), "The script is required to return a valid VTK or Pyvista object")
        persistent_storage["nodes"][self.name] = result

        pass

    def get_vtkobj(self):
        if self.name in persistent_storage["nodes"]:
            return persistent_storage["nodes"][self.name]
        else:
            return pv.PolyData() #Return an empty. This is necessary for BVTK to actually call apply properties

    def get_output(self, socketname):
        '''Execute user defined function. If something goes wrong,
        print the error and return the input object.
        '''
        return self.get_vtkobj()

class BVTK_Node_PyvistaCalculator(PyvistaComputeHelper, Node, BVTK_Node):
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

    def apply_properties(self, vtkobj):
        print("Apply properties called with " + str(vtkobj))
        assert_bvtk(type(vtkobj) in vtk_pv_mapping.values(), "Expected a pyvista type. This is an internal error... try updating")

        (data_dicts, pvobjs, vtkobjs) = self.get_data_dicts()

        #Choose the correct data dictionary
        if self.e_AttrType == 'PointData':
            data_dicts = data_dicts['point_arrays']
        elif self.e_AttrType == 'CellData':
            data_dicts = data_dicts['cell_arrays']
        else:
            data_dicts = data_dicts['field_arrays']

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

        assert_bvtk(isinstance(result, np.ndarray) and result.ndim > 0 and (expected_size is None or result.shape[0] == expected_size), 
                        "Expected a [%d (, x)] array as a result of the lambda expression. Got %s" % (expected_size, result))
        target_dict[self.m_ResultName] = result #Change the mutable dictionary in-place
        persistent_storage["nodes"][self.name] = vtkobj

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

    #def get_vtkobj(self):
    #    return self.get_output("output")

    def get_output(self, socketname):
        return self.get_input_node('input')[0]


edge_count_vtk_type_map = {2: vtk.VTK_LINE, 3: vtk.VTK_TRIANGLE, 4: vtk.VTK_QUAD, 8: vtk.VTK_HEXAHEDRON}

def mean_cell_to_point(arr, cell_map, nr_points):
    counts = np.zeros(shape=[nr_points], dtype=arr.dtype)
    mean_arr = counts.copy()
    np.add.at(counts, cell_map, np.ones_like(arr))
    np.add.at(mean_arr, cell_map, arr)
    return mean_arr / counts

class BVTK_Node_BlenderToVTK(Node, BVTK_Node):
    '''BVTK BlenderToVTK Node'''
    bl_idname = 'BVTK_Node_BlenderToVTKType'
    bl_label  = 'BlenderToVTK'

    input_obj_prop: bpy.props.PointerProperty(type=bpy.types.Object)
    #test_prop: bpy.props.PointerProperty(type=bpy.types.PropertyGroup)
    output_type_items = [ (x,x,x) for x in ['UnstructuredGrid', 'PolyData']]
    output_type_prop:   bpy.props.EnumProperty   (name='Output Type', default='UnstructuredGrid', items=output_type_items)
    #triangulate: bpy.props.BoolProperty(name='Triangulate', default=True)
    apply_transforms: bpy.props.BoolProperty(name='Apply Transformations', 
                            description='Apply the objects transformations (Translation, Rotation and Scale) before conversion', 
                            default=True)
    apply_modifiers: bpy.props.BoolProperty(name='Apply Modifiers', 
                            description='Apply the objects modifiers (if activated in Preview) before conversion.', 
                            default=True)
    copy_normals: bpy.props.BoolProperty(name='Copy Normals', default=True)
    copy_vertex_cols: bpy.props.BoolProperty(name='Copy Vertex Colors', default=True)
    b_properties: bpy.props.BoolVectorProperty(name="", size=6, get=BVTK_Node.get_b, set=BVTK_Node.set_b)

    def m_properties(self):
        return ['input_obj_prop', 'output_type_prop', 'apply_transforms', 'apply_modifiers', 'copy_normals', 'copy_vertex_cols']

    def m_connections(self):
        return ([],[],[],['output'])

    #No inputs to process
    def apply_inputs(self, vtkobj_ignored):
        pass

    def apply_properties(self, vtkobj_ignored):
        #No mesh selected
        if self.input_obj_prop is None:
            return

        obj = self.input_obj_prop
        assert_bvtk(type(obj.data) == bpy.types.Mesh, "Could not convert Object " + obj.name + " to VTK. No underlying mesh found")

        #https://blender.stackexchange.com/questions/146559/how-do-i-get-a-mesh-data-block-with-modifiers-and-shape-keys-applied-in-blender
        if self.apply_modifiers:
            dg = bpy.context.evaluated_depsgraph_get()
            obj = obj.evaluated_get(dg)
        
        mesh = obj.data
            

        #Read vertices
        nr_points = len(mesh.vertices)
        points = np.zeros(shape=[nr_points * 3])
        mesh.vertices.foreach_get("co", points)
        points = points.reshape([-1, 3])

        if self.apply_transforms:
            points_homogeneous = np.ones(shape=[points.shape[0], 4])
            points_homogeneous[..., :-1] = points
            transform_mat = obj.matrix_world
            transform_mat = np.array([transform_mat.row[i][:] for i in range(4)])
            points_transformed = np.einsum('xy,...y->...x', transform_mat, points_homogeneous)
            points = points_transformed[..., :-1] / points_transformed[..., -1:]


        nr_faces = len(mesh.polygons)
        faces_size = mesh.polygons[-1].loop_indices[-1] + 1
        faces = np.zeros(shape=[faces_size], dtype=np.int32)
        face_sizes = np.zeros(shape=[nr_faces], dtype=np.int32)
        mesh.polygons.foreach_get("vertices", faces)
        mesh.polygons.foreach_get("loop_total", face_sizes)
        face_offsets = np.concatenate([[0], np.cumsum(face_sizes)[:-1]])
        vtk_faces = np.insert(faces, face_offsets, face_sizes)

        if self.output_type_prop == "UnstructuredGrid":
            cell_types = np.array([(edge_count_vtk_type_map[size] if size in edge_count_vtk_type_map else vtk.VTK_POLYGON) for size in face_sizes])

            vtk_mesh = pv.UnstructuredGrid(vtk_faces, cell_types, points)
        elif self.output_type_prop == "PolyData":
            vtk_mesh = pv.PolyData(points, vtk_faces)

        if self.copy_vertex_cols:
            for vcol_lay in mesh.vertex_colors:
                colors_and_alpha = np.zeros(shape=[faces_size * 4], dtype=np.float32)
                vcol_lay.data.foreach_get("color", colors_and_alpha)
                colors_and_alpha = colors_and_alpha.reshape([-1, 4])
                vtk_mesh.point_arrays["vertex_color_" + vcol_lay.name] = np.stack([mean_cell_to_point(colors_and_alpha[..., i], faces, nr_points) for i in range(3)], axis=-1) #Alpha channel is ignored

        #TODO: Point or cell normals?
        if self.copy_normals:
            point_normals = np.zeros(shape=[nr_points * 3], dtype=np.float32)
            mesh.vertices.foreach_get("normal", point_normals)
            point_normals = point_normals.reshape([-1, 3])
            vtk_mesh.point_arrays["normals"] = point_normals

        #if self.triangulate:
        #    vtk_mesh = vtk_mesh.triangulate()

        persistent_storage["nodes"][self.name] = vtk_mesh

    def get_output(self, socketname):
        return None

    def get_vtkobj(self):
        return self.get_output("output")

    def get_output(self, socketname):
        if self.input_obj_prop is None:
            return None

        elif self.name in persistent_storage["nodes"]:
            return persistent_storage["nodes"][self.name]
        elif self.output_type_prop == "UnstructuredGrid":
            return pv.UnstructuredGrid()
        elif self.output_type_prop == "PolyData":
            return pv.PolyData()
        else:
            assert_bvtk(False, "Unexpected output type for " + self.name)


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
