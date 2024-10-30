import numpy as np
import dolfinx
import basix
from dolfinx.io import gmshio
from mpi4py import MPI

import opensg

class BladeMesh:
    """This class processes and stores information about a wind turbine blade's mesh
    """
    def __init__(self, mesh_data):
        """

        Parameters
        ----------
        mesh_data : dict
            dictionary of mesh data loaded in from the output of pynumad mesher
        """
        self._mesh_data = mesh_data
        
        self.nodes = mesh_data['nodes']
        self.num_nodes = len(self.nodes)
        
        self.elements = mesh_data['elements']
        self.num_elements = len(self.elements)
        
        self.sets = mesh_data["sets"]
        self.materials = mesh_data["materials"]        
        self.sections = mesh_data["sections"]
        self.element_orientations = mesh_data["elementOrientations"]
        
        self._generate_material_database()
        
        
    def _generate_material_database(self):
        material_database = dict()
        
        for i, material in enumerate(self.materials):
            material_dict = dict()
            
            material_dict["id"] = i
            
            elastic = material['elastic']
            material_dict["E"] = elastic['E']
            material_dict["G"] = elastic['G']
            material_dict["nu"] = elastic['nu']
            
            material_database[material['name']] = material_dict
        
        self.material_database = material_database
        return
    
        
    def generate_segment_mesh(self, segment_index, filename):
        segment_node_labels = -1 * np.ones(self.num_nodes, dtype=int)
        segment_element_labels = -1 * np.ones(self.num_elements, dtype=int)
        segment_element_layer_id = -1 * np.ones(self.num_elements, dtype=int)

        layer_count = 0 # NOTE: Are we assuming that element sets are ordered in a particular way? -klb
        for element_set in self.sets["element"]:
            name = element_set["name"]
            name_components = name.split("_")
            
            labels = element_set["labels"]
            if len(name_components) > 2:
                if (int(name_components[1]) == segment_index):
                    for element_label in labels:
                        segment_element_labels[element_label] = 1
                        segment_element_layer_id[element_label] = layer_count
                        for node in self.elements[element_label]:
                            if (node > -1): # when would this not be the case? - klb
                                segment_node_labels[node] = 1
                    layer_count += 1

        element_label = 1
        for i, e in enumerate(segment_element_labels):
            if (e == 1):
                segment_element_labels[i] = element_label
                element_label += 1

        node_label = 1
        for i, n in enumerate(segment_node_labels):
            if (n == 1):
                segment_node_labels[i] = node_label
                node_label += 1
                    
        file = open(filename,'w')

        file.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n')
        newNumNds = np.max(segment_node_labels)
        file.write(str(newNumNds) + '\n')

        for i, nd in enumerate(self.nodes):
            lab = segment_node_labels[i]
            if(lab > -1):
                ln = [str(lab),str(nd[2]),str(nd[0]),str(nd[1])]
            #  ln = [str(lab),str(nd[0]),str(nd[1]),str(nd[2])]
                file.write(' '.join(ln) + '\n')

        file.write('$EndNodes\n$Elements\n')

        newNumEls = np.max(segment_element_labels)
        file.write(str(newNumEls) + '\n')

        for i, el in enumerate(self.elements):
            lab = segment_element_labels[i]
            if(lab > -1):
                ln = [str(lab)]
                if(el[3] == -1):
                    ln.append('2')
                else:
                    ln.append('3')
                ln.append('2')
                ln.append(str(segment_element_layer_id[i]+1))
                ln.append(str(segment_element_layer_id[i]+1))
                for nd in el:
                    if(nd > -1):
                        ln.append(str(segment_node_labels[nd]))
                file.write(' '.join(ln) + '\n')
        file.write('$EndElements\n')

        file.close()
        
        # initialize segmentmesh object
        segment_mesh = SegmentMesh(
            segment_node_labels=segment_node_labels,
            segment_element_labels=segment_element_labels,
            segment_element_layer_id=segment_element_layer_id,
            segment_index=segment_index, 
            parent_blade_mesh=self, 
            msh_file=filename)
        return segment_mesh
        
        
class SegmentMesh():
    def __init__(
        self,
        segment_node_labels,
        segment_element_labels,
        segment_element_layer_id,
        segment_index, 
        parent_blade_mesh, 
        msh_file):
        """This class manages the data and methods for the mesh of a segment of a blade.
        
        A segment is defined as the part of the blade between two fixed points along the blade span.
        Given a set of N span points along the blade, there are N-1 segments defined between each consecutive pair
        of span points. For example, the segment indexed by 0 is defined between the span points indexed by 0 and 1

        Parameters
        ----------
        segment_node_labels : array[int]
            _description_
        segment_element_labels : array[int]
            _description_
        segment_element_layer_id : array[int]
            _description_
        segment_index : int
            Index of the segment of blade
        parent_blade_mesh : BladeMesh
            BladeMesh object that SegmentMesh derives from.
        msh_file : str or Path
            Path to mesh file to load data from
        """
        
        self.segment_node_labels = segment_node_labels
        self.segment_element_labels = segment_element_labels
        self.segment_element_layer_id = segment_element_layer_id
        self.segment_index = segment_index
        self.blade_mesh = parent_blade_mesh
        
        self.mesh, self.subdomains, self.boundaries = gmshio.read_from_msh(msh_file, MPI.COMM_WORLD,0, gdim=3)
        self.original_cell_index = self.mesh.topology.original_cell_index # Original cell Index from mesh file
        lnn = self.subdomains.values[:]-1
        self.num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local 
        cells = np.arange(self.num_cells, dtype=np.int32)
        # NOTE: Unsure what this next line does.
        self.subdomains = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim, cells, np.array(lnn,dtype=np.int32))
        
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
        
        return
    
    def _generate_layup_data(self):
        layup_database = dict()
        
        mat_names, thick, angle, nlay = [], [], [], []
        for section in self.blade_mesh.sections:
            name_components = section['elementSet'].split('_')
            if(len(name_components) > 2):
                material_name, t, an = [], [], []
                if(int(name_components[1]) == self.segment_index):
                    layup = section['layup'] # layup = [[material_name: str, thickness: float, angle:]]
                    nlay.append(len(layup))
                    for layer in layup:
                        material_name.append(layer[0])     
                    mat_names.append(material_name)
                    for layer in layup:
                        t.append(layer[1])
                    thick.append(t)
                    for layer in layup:
                        an.append(layer[2])
                    angle.append(an) 
        
        layup_database["mat_names"] = mat_names
        layup_database["thick"] = thick
        layup_database["angle"] = angle
        layup_database["nlay"] = nlay
        self.layup_database = layup_database
        
        return layup_database
    

    def generate_local_orientations(self):
        # Local Orientation (DG0 function) of quad mesh element (from yaml data)
        VV = dolfinx.fem.functionspace(
            self.mesh, basix.ufl.element(
            "DG", self.mesh.topology.cell_name(), 
            0, shape=(3, )))
        EE1, EE2, N = dolfinx.fem.Function(VV), dolfinx.fem.Function(VV), dolfinx.fem.Function(VV) 

        orientations = []
        for i, eo in enumerate(self.blade_mesh.element_orientations):
            if(self.segment_element_labels[i] > -1):
                o = []
                for k in range(9):
                    o.append(eo[k])
                orientations.append(o)

        # Store orientation for each element
        # TODO: clarify variable names. why N?
        for k, ii in enumerate(self.original_cell_index):
        # Storing data to DG0 functions 
            EE2.x.array[3*k], EE2.x.array[3*k+1], EE2.x.array[3*k+2] = orientations[ii][5], orientations[ii][3], orientations[ii][4]  # e2
            N.x.array[3*k], N.x.array[3*k+1], N.x.array[3*k+2] = orientations[ii][8], orientations[ii][6], orientations[ii][7]   #  e3 
            EE1.x.array[3*k], EE1.x.array[3*k+1], EE1.x.array[3*k+2] = orientations[ii][2], orientations[ii][0], orientations[ii][1]  # e1    outward normal 
        self.EE1 = EE1
        self.N = N
        self.EE2 = EE2
        frame = [EE1,EE2,N]
        return frame
    
    
    def extract_boundaries(self):
        # extract geometry
        pp = self.mesh.geometry.x

        is_left_boundary, is_right_boundary = opensg.util.generate_boundary_markers(
            min(pp[:,0]), max(pp[:,0]))
        

        facets_left = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_left_boundary)
        facets_right = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_right_boundary)

        right_mesh, right_entity_map, right_vertex_map, right_geom_map = dolfinx.mesh.create_submesh(
            self.mesh, self.fdim, facets_right)
        left_mesh, left_entity_map, left_vertex_map, left_geom_map = dolfinx.mesh.create_submesh(
            self.mesh, self.fdim, facets_left)
        
        self.right_submesh = {
            "mesh": right_mesh, 
            "entity_map": right_entity_map, 
            "vertex_map": right_vertex_map, 
            "geom_map": right_geom_map}
        
        self.left_submesh = {
            "mesh": left_mesh, 
            "entity_map": left_entity_map, 
            "vertex_map": left_vertex_map, 
            "geom_map": left_geom_map}
    
        # generate subdomains
        self.mesh.topology.create_connectivity(2,1)  # (quad mesh topology, boundary(1D) mesh topology)
        cell_of_facet_mesh = self.mesh.topology.connectivity(2,1)
        
        # Cell to Edge connectivity
        # cell_edge_map = []
        # for i in range(self.num_cells):
        #     c = []
        #     for k in range(4): # 4 is used as number of edges in a quad element
        #         c.append((cell_of_facet_mesh.array[4*i+k])) 
        #     cell_edge_map.append(c)
        # cell_edge_map = np.ndarray.flatten(np.array(cell_edge_map))
        
        # 
        def subdomains_boundary(boundary_mesh, boundary_marker, boundary_entity_map):
            boundary_VV = dolfinx.fem.functionspace(
                boundary_mesh, basix.ufl.element("DG", boundary_mesh.topology.cell_name(), 0, shape=(3, )))
            
            boundary_e1 = dolfinx.fem.Function(boundary_VV)
            boundary_e2 = dolfinx.fem.Function(boundary_VV)
            boundary_n = dolfinx.fem.Function(boundary_VV)
            
            boundary_subdomains = []
            
            boundary_facets_left = dolfinx.mesh.locate_entities(boundary_mesh, self.fdim, boundary_marker)

            # TODO: review the subdomain assingments with akshat
            for i, xx in enumerate(boundary_entity_map):
                # assign subdomain
                idx = int(np.where(cell_of_facet_mesh.array==xx)[0]/4) # 4 is for number of nodes in quad element
                boundary_subdomains.append(self.subdomains.values[idx])
                # assign orientation
                for j in range(3):
                    boundary_e1.x.array[3*i+j] = self.EE1.x.array[3*idx+j]
                    boundary_e2.x.array[3*i+j] = self.EE2.x.array[3*idx+j]
                    boundary_n.x.array[3*i+j] = self.N.x.array[3*idx+j]

            frame = [boundary_e1, boundary_e2, boundary_n]
            boundary_subdomains = np.array(boundary_subdomains, dtype=np.int32)
            boundary_num_cells = boundary_mesh.topology.index_map(boundary_mesh.topology.dim).size_local 
            boundary_cells = np.arange(boundary_num_cells, dtype=np.int32)
            boundary_subdomains = dolfinx.mesh.meshtags(boundary_mesh, boundary_mesh.topology.dim, boundary_cells, boundary_subdomains)
            return boundary_subdomains, frame, boundary_facets_left
            # Mapping the orinetation data from quad mesh to boundary. The alternative is to use local_frame_1D(mesh_l).
            # Either of both can be used in local_boun subroutine 
        
        self.left_submesh["subdomains"], self.left_submesh["subdomains"], self.left_submesh["subdomains"] = subdomains_boundary(
            self.left_submesh["mesh"], is_left_boundary, self.left_submesh["entity_map"]) 
        self.right_submesh["subdomains"], self.right_submesh["subdomains"], self.right_submesh["subdomains"] = subdomains_boundary(
            self.right_submesh["mesh"], is_right_boundary, self.right_submesh["entity_map"])

    def generate_boundary_ABD(self):
        nphases = max(self.subdomains.values[:]) + 1
        ABD_ = []
        for i in range(nphases):
            ABD_.append(opensg.ABD_mat(
                i, 
                thick=self.layup_database["thick"], 
                nlay=self.layup_database["nlay"], 
                mat_names=self.layup_database["mat_names"],
                angle=self.layup_database["angle"],
                material_database=self.blade_mesh.material_database
                ))
            
        # print('Computed',nphases,'ABD matrix')

        # def ABD_matrix(i):
        #     return(as_tensor(ABD_[i]))