from .core import l # Import logging
from .core import *
from .converters import with_pyvista

if with_pyvista:
    from .custom_nodes.pynodes.pynodes import vtk_pv_mapping, is_mappable_pyvista_type, is_pyvista_obj, map_vtk_to_pv_obj
    import numpy as np

class BVTK_Node_Info(Node, BVTK_Node):
    '''BVTK Info Node'''
    bl_idname = 'BVTK_Node_InfoType'
    bl_label  = 'Info'

    arr_string = '{k} [{i}] ({data_type_name}{n_comps}): \'{name}\': {range_text}'
    active_component_string = 'Active {comp}: {comp_name}'

    def m_properties(self):
        return []

    def m_connections(self):
        return (['input'],[],[],['output'])

    def update_cb(self):
        l.debug('tree updated')

    def update(self):
        # Make info node wider to show all text
        self.width = 300

    def draw_buttons(self, context, layout):
        fs="{:.5g}" # Format string
        in_node, vtkobj = self.get_input_node('input')
        if not in_node:
            layout.label(text='Connect a node')
        elif not vtkobj:
            layout.label(text='Input has no vtkobj (try updating)')
        else:
            vtkobj = resolve_algorithm_output(vtkobj)
            if not vtkobj:
                layout.label(text='Failed to resolve algorithm ouput (try updating)')
            else:
                layout.label(text='Type: ' + vtkobj.__class__.__name__)
                if hasattr(vtkobj, 'GetNumberOfPoints'):
                    layout.label(text='Points: ' + str(vtkobj.GetNumberOfPoints()))
                if hasattr(vtkobj, 'GetNumberOfCells'):
                    layout.label(text='Cells: ' + str(vtkobj.GetNumberOfCells()))
                if hasattr(vtkobj, 'GetBounds'):
                    layout.label(text='X range: ' + fs.format(vtkobj.GetBounds()[0]) + \
                                ' - ' + fs.format(vtkobj.GetBounds()[1]))
                    layout.label(text='Y range: ' + fs.format(vtkobj.GetBounds()[2]) + \
                                ' - ' + fs.format(vtkobj.GetBounds()[3]))
                    layout.label(text='Z range: ' + fs.format(vtkobj.GetBounds()[4]) + \
                                ' - ' + fs.format(vtkobj.GetBounds()[5]))
                data = {}
                if hasattr(vtkobj, 'GetPointData'):
                    data['Point data'] = vtkobj.GetPointData()
                if hasattr(vtkobj, 'GetCellData'):
                    data['Cell data'] = vtkobj.GetCellData()
                if hasattr(vtkobj, 'GetFieldData'):
                    data['Field data'] = vtkobj.GetFieldData()
                for k in data:
                    d = data[k]
                    for i in range(d.GetNumberOfArrays()):
                        arr = d.GetArray(i)
                        data_type_name = arr.GetDataTypeAsString()
                        n_comps = arr.GetNumberOfComponents()
                        name = arr.GetName()
                        
                        if name is None or data_type_name is None or n_comps is None:
                            l.warning("Invalid array encountered...")

                        range_text = ''
                        for n in range(n_comps):
                            r = arr.GetRange(n)
                            range_text += '[' + fs.format(r[0]) +', ' + fs.format(r[1]) + ']  '
                        row = layout.row()
                        row.label(text = self.arr_string.format(k=k, i=i, data_type_name=data_type_name, 
                            n_comps=n_comps, name=name, range_text=range_text))

            #TODO: Takes a lot of computations since there is no caching for this part
            #      Either use a cache or an optional flag for this
            if with_pyvista:
                pvobj = None
                if is_pyvista_obj(vtkobj):
                    pvobj = vtkobj

                elif is_mappable_pyvista_type(vtkobj):
                    pvobj = map_vtk_to_pv_obj(vtkobj)(vtkobj)

                if pvobj is not None:
                    #Show the active arrays (scalars, vectors, tensors)
                    for comp in ["scalars", "vectors", "tensors"]:
                        active_component_name = getattr(pvobj, "active_" + comp + "_name")
                        if active_component_name is not None:
                            row = layout.row()
                            row.label(text = self.active_component_string.format(comp=comp.capitalize(), comp_name=active_component_name))

        layout.separator()
        row = layout.row()
        row.separator()
        row.separator()
        row.separator()
        row.separator()
        row.operator("node.bvtk_node_update", text="update").node_path = node_path(self)
        row.separator()
        row.separator()
        row.separator()
        row.separator()

    def apply_properties(self, vtkobj):
        pass

    def get_output(self, socketname):
        return self.get_input_node('input')[1]

TYPENAMES = []
add_class(BVTK_Node_Info)
TYPENAMES.append('BVTK_Node_InfoType')

menu_items = [NodeItem(x) for x in TYPENAMES]
CATEGORIES.append(BVTK_NodeCategory("Debug", "Debug", items=menu_items))
