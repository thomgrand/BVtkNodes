from ...update import *
from ...core import l # Import logging
from ...core import *
import bmesh
import bpy
import numpy as np
import pyvista as pv
from ...errors.bvtk_errors import BVTKException, assert_bvtk


class ArbitraryInputsHelper():
    '''Helps in organizing nodes with an arbitrary number of input nodes that will
    occur and disappear according to the connections made. See Pyvista-Calculator Node
    for an example
    '''
    input_limit = 10
    input_format_str = "Input {}"

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

class BVTK_Node_PyvistaFilter(Node, BVTK_Node):
    pass

class BVTK_Node_PyvistaCalculator(ArbitraryInputsHelper, Node, BVTK_Node):
    bl_idname = 'BVTK_Node_PyvistaCalculatorType' # type name
    bl_label  = 'Pyvista Calculator' # label for nice name display

    m_LambdaCode:   bpy.props.StringProperty (name='Execution Code', default='')

    e_AttrType_items = [ (x,x,x) for x in ['PointData', 'CellData', 'FieldData']]

    m_ResultName:   bpy.props.StringProperty (name='Result Name', default='result')
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
        vtkobj.deep_copy(pvobjs[0]) #Take shape and arguments from the input

        lambda_eval_str = "lambda: " + self.m_LambdaCode

        #Only one input: Unpack dictionary
        if len(data_dicts) == 1:
            local_dict = {k: v for k, v in data_dicts[0].items()}
            local_dict["numpy"] = np
            local_dict["np"] = np
            try:
                result = eval(lambda_eval_str, local_dict)()
            except Exception as ex:
                err_msg = "Execution of the given calculator function failed with: {}".format(ex)
                l.error(err_msg)
                self.m_InfoMsg = err_msg
                return self.get_input_node('input')[1]
    
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

        else:
            assert(False) #self.create_lambda_locals(data_dicts)

        pass
    
    """
    def update(self):
        super(BVTK_Node_PyvistaCalculator, self).update()
        vtkobjs = self.get_input_vtkobjs()
        if self.input_is_invalid(vtkobjs):
            return

        (data_dicts, pvobjs, vtkobjs) = self.get_data_dicts()
        self.outputs["output"].data = pvobjs[0].copy()
    """

    def apply_inputs(self, vtkobj):
        print("Apply input called with " + str(vtkobj))
        pass

    def input_is_invalid(self, vtkobjs):
        return len(vtkobjs) == 0 or vtkobjs[0] is None or vtkobjs[0] == 0

    def get_vtkobj(self):
        vtkobjs = self.get_input_vtkobjs()
        if self.input_is_invalid(vtkobjs):
            return None
        
        (data_dicts, pvobjs, vtkobjs) = self.get_data_dicts(vtkobjs)
        return pvobjs[0]

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
        #(data_dicts, pvobjs, vtkobjs) = self.get_data_dicts(vtkobjs)
        return vtkobjs[0]
        #return vtkobjs[0]

    def get_input_vtkobjs(self):
        nr_inputs = self.current_active_inputs
        vtkobjs = []

        for input_i in range(nr_inputs):
            in_node, vtkobj = self.get_input_node(self.inputs[input_i].name)
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

    def free(self):
        if self.name in persistent_storage["nodes"]:
            del persistent_storage["nodes"][self.name]

#class BVTK_Node_SetActiveArrays(Node, BVTK_Node):
#    '''Convenience node that helps in setting the active scalars, vectors or tensors
#    '''
#    pass


#def register_nodes():
l.info("Registering Pyvista nodes")
# Add classes and menu items
TYPENAMES = []
add_class(BVTK_Node_PyvistaSource)
TYPENAMES.append('BVTK_Node_PyvistaSource')
add_class(BVTK_Node_ArbitraryInputsTest)
TYPENAMES.append('BVTK_Node_ArbitraryInputsTestType')
add_class(BVTK_Node_PyvistaCalculator)
TYPENAMES.append('BVTK_Node_PyvistaCalculatorType')
"""
add_class(BVTK_Node_PyvistaFilter)
TYPENAMES.append('BVTK_Node_PyvistaFilter')
add_class(BVTK_Node_PyvistaCalculator)
TYPENAMES.append('BVTK_Node_PyvistaCalculator')
"""
menu_items = [NodeItem(x) for x in TYPENAMES]
CATEGORIES.append(BVTK_NodeCategory("Pyvista", "Pyvista", items=menu_items))
