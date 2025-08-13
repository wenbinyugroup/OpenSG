from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
import dolfinx
import basix
from ufl import dot, as_tensor, rhs
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import form, petsc, Function, locate_dofs_topological, apply_lifting, set_bc
from scipy.sparse import csr_matrix
import scipy

import opensg
from opensg.core import shell as core
from ..utils import shell as utils


class ShellBladeMesh:
    """A class representing the complete mesh of a wind turbine blade.
    
    This class processes and stores information about a wind turbine blade's mesh,
    including nodes, elements, material properties, and section definitions. It provides
    methods to generate segment meshes and manage material properties.
    
    Attributes
    ----------
    nodes : list
        List of node coordinates for the blade mesh
    num_nodes : int
        Total number of nodes in the mesh
    elements : list
        List of element definitions
    num_elements : int
        Total number of elements in the mesh
    sets : dict
        Dictionary containing element and node sets
    materials : dict
        Dictionary containing material definitions
    sections : dict
        Dictionary containing section definitions
    element_orientations : list
        List of element orientation matrices
    material_database : dict
        Processed database of material properties
    """
    def __init__(self, mesh_data):
        """Initialize a BladeMesh object from mesh data.

        Parameters
        ----------
        mesh_data : dict
            Dictionary of mesh data loaded from the output of pynumad mesher.
            Expected to contain:
                - nodes: list of node coordinates
                - elements: list of element definitions
                - sets: dictionary of element and node sets
                - materials: dictionary of material definitions
                - sections: dictionary of section definitions
                - elementOrientations: list of element orientation matrices
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
        """Generate a processed database of material properties.
        
        This internal method processes the raw material data into a more
        accessible format. For each material, it extracts:
            - Material ID
            - Elastic properties (E, G, nu)
            
        The processed data is stored in self.material_database.
        
        Returns
        -------
        None
        """
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
    
        
    def generate_segment_mesh(self, segment_index, filename="segment_mesh.msh"):
        """Generate a mesh for a specific blade segment.
        
        This method extracts a portion of the blade mesh corresponding to the
        specified segment index and creates a new SegmentMesh object. The segment
        mesh is also written to a file in GMSH format.
        
        Parameters
        ----------
        segment_index : int
            Index of the blade segment to extract
        filename : str
            Name of the file to write the GMSH mesh to
            
        Returns
        -------
        SegmentMesh
            A new SegmentMesh object containing the extracted segment data
        """
        segment_node_labels = -1 * np.ones(self.num_nodes, dtype=int)
        segment_element_labels = -1 * np.ones(self.num_elements, dtype=int)
        segment_element_layer_id = -1 * np.ones(self.num_elements, dtype=int)
        
        # Load Layup Data 
        mat_names, thick,angle,nlay=[],[],[],[]
        for sec in self.sections:
            name_components = sec['elementSet'].split('_')
            if(len(name_components) > 2):
                m,t,an=[],[],[]
                if(int(name_components[1]) == segment_index):
                    layup = sec['layup']
                    nlay.append(len(layup))
                    for l in layup:
                        m.append(l[0])     
                    mat_names.append(m)
                    for l in layup:
                        t.append(l[1])
                    thick.append(t)
                    for l in layup:
                        an.append(l[2])
                    angle.append(an) 
        combined_rows = list(zip(thick, mat_names, angle))

        ii = 0
        unique_rows=[]
        for es in self.sets['element']:
            name_components = es['name'].split('_')
            if(len(name_components) > 2):
                # Some section names do not have indices. These are assumed to represent
                # groups of multiple sections (eg all of the shear web elements)
                # and are ignored.
                try:
                    section_index = int(name_components[1])
                except ValueError:
                    continue
                
                if(section_index == segment_index):
                    if combined_rows[ii] not in unique_rows:
                    #   layCt+=1
                        unique_rows.append(combined_rows[ii])

                    lay_num=unique_rows.index(combined_rows[ii])    
                    for eli in es['labels']:
                        segment_element_labels[eli] = 1
                        segment_element_layer_id[eli] = lay_num
                        for nd in self.elements[eli]:
                            if(nd > -1):
                                segment_node_labels[nd] = 1
                    ii+=1

        thick, mat_names, angle = list(zip(*unique_rows))[0], list(zip(*unique_rows))[1], list(zip(*unique_rows))[2]
        nlay = [len(item) for item in thick]
        # combine into layup dictionary
        layup_database = {"mat_names": mat_names, "thick": thick, "angle": angle, "nlay": nlay}
        
        
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
        segment_mesh = BladeSegmentMesh(
            segment_node_labels=segment_node_labels,
            segment_element_labels=segment_element_labels,
            segment_element_layer_id=segment_element_layer_id,
            segment_index=segment_index,
            layup_database=layup_database,
            parent_blade_mesh=self, 
            msh_file=filename)
        return segment_mesh


class SolidBladeSegmentMesh():
    def __init__(
        self,
       # segment_node_labels,
       # segment_element_labels,
      #  segment_element_layer_id,
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

     #   self.segment_node_labels = segment_node_labels
      #  self.segment_element_labels = segment_element_labels
      #  self.segment_element_layer_id = segment_element_layer_id
        self.segment_index = segment_index
        self.blade_mesh = parent_blade_mesh
        self.elLayID=parent_blade_mesh.elLayID
        self.material_database=parent_blade_mesh.material_database
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)    # mesh read output will not be printed
        self.mesh, self.subdomains, self.boundaries = gmshio.read_from_msh(msh_file, MPI.COMM_WORLD,0, gdim=3)
        self.original_cell_index = self.mesh.topology.original_cell_index # Original cell Index from mesh file
   #     lnn = self.subdomains.values[:]-1
     #   self._generate_layup_id()

        lnn=[]
        for k in self.original_cell_index:
             lnn.append(self.elLayID[k])

        self.num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        cells = np.arange(self.num_cells, dtype=np.int32)

        self.subdomains = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim, cells, np.array(lnn,dtype=np.int32))

        # Update elLayID to contain the mapped segment layup IDs (same as StandaloneSolidSegmentMesh)
        # self.elLayID = np.array(lnn, dtype=np.float64)

        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1

      #  self._generate_layup_data()
   #     self._generate_material_database()
        self._build_local_orientations()
        self._build_boundary_submeshes()

        return

    def _build_local_orientations(self):
        # Local Orientation (DG0 function) of quad mesh element (from yaml data)
        VV = dolfinx.fem.functionspace(
            self.mesh, basix.ufl.element(
            "DG", self.mesh.topology.cell_name(),
            0, shape=(3, )))
        EE1, EE2, EE3 = dolfinx.fem.Function(VV), dolfinx.fem.Function(VV), dolfinx.fem.Function(VV)

        orientations = []
        for i, eo in enumerate(self.blade_mesh.element_orientations):
                o = []
                for k in range(9):
                    o.append(eo[k])
                orientations.append(o)

        # Store orientation for each element
        # TODO: clarify variable names. why N?
        for k, ii in enumerate(self.original_cell_index):
        # Storing data to DG0 functions 
            EE2.x.array[3*k], EE2.x.array[3*k+1], EE2.x.array[3*k+2] = orientations[ii][5], orientations[ii][3], orientations[ii][4]  # e2
            EE3.x.array[3*k], EE3.x.array[3*k+1], EE3.x.array[3*k+2] = orientations[ii][8], orientations[ii][6], orientations[ii][7]   #  e3 
            EE1.x.array[3*k], EE1.x.array[3*k+1], EE1.x.array[3*k+2] = orientations[ii][2], orientations[ii][0], orientations[ii][1]  # e1    outward normal 
        self.EE1 = EE1
        self.EE3 = EE3
        self.EE2 = EE2
        frame = [EE1,EE2,EE3]
        self.frame = frame
        return frame

    def _build_boundary_submeshes(self):
        pp = self.mesh.geometry.x
        x_min,x_max=min(pp[:,0]), max(pp[:,0])
        mean=0.5*(x_min+x_max)  # Mid origin for taper segments

        blade_length=x_max-x_min

        self.left_origin,self.right_origin,self.taper_origin=[],[],[]
        self.left_origin.append(float(x_min)/blade_length),self.right_origin.append(float(x_max)/blade_length),self.taper_origin.append(float(mean)/blade_length)
       # print(float(mean))
    #    pp[:,0]=pp[:,0]-mean

        is_left_boundary, is_right_boundary = opensg.utils.solid.generate_boundary_markers(
            min(pp[:,0]), max(pp[:,0]))

        left_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_left_boundary)
        right_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_right_boundary)

        left_mesh, left_entity_map, left_vertex_map, left_geom_map = dolfinx.mesh.create_submesh(
            self.mesh, self.fdim, left_facets)
        right_mesh, right_entity_map, right_vertex_map, right_geom_map = dolfinx.mesh.create_submesh(
            self.mesh, self.fdim, right_facets)

        self.left_submesh = {
            "mesh": left_mesh,
            "entity_map": left_entity_map,
            "vertex_map": left_vertex_map,
            "geom_map": left_geom_map,
            "marker": is_left_boundary}
            # "facets": left_facets}

        self.right_submesh = {
            "mesh": right_mesh,
            "entity_map": right_entity_map,
            "vertex_map": right_vertex_map,
            "geom_map": right_geom_map,
            "marker": is_right_boundary}
            # "facets": right_facets}

        self.mesh.topology.create_connectivity(3,2)  # (quad mesh topology, boundary(1D) mesh topology)
        cell_of_facet_mesh = self.mesh.topology.connectivity(3,2)

        # NOTE: found a different way to determine connectivity that didn't need this block -klb
        # Cell to Edge connectivity
        # cell_edge_map = []
        # for i in range(self.num_cells):
        #     c = []
        #     for k in range(4): # 4 is used as number of edges in a quad element
        #         c.append((cell_of_facet_mesh.array[4*i+k])) 
        #     cell_edge_map.append(c)
        # cell_edge_map = np.ndarray.flatten(np.array(cell_edge_map))

        # generate subdomains
        def _build_boundary_subdomains(boundary_meshdata):
            boundary_mesh = boundary_meshdata["mesh"]
            boundary_entity_map = boundary_meshdata["entity_map"]
            boundary_marker = boundary_meshdata["marker"]
            boundary_VV = dolfinx.fem.functionspace(
                boundary_mesh, basix.ufl.element("DG", boundary_mesh.topology.cell_name(), 0, shape=(3, )))

            boundary_e1 = dolfinx.fem.Function(boundary_VV)
            boundary_e2 = dolfinx.fem.Function(boundary_VV)
            boundary_e3 = dolfinx.fem.Function(boundary_VV)

            boundary_facets = dolfinx.mesh.locate_entities(boundary_mesh, self.fdim, boundary_marker)

            # TODO: review the subdomain assingments with akshat
            boundary_subdomains = []
            el_facets=6
            for i, xx in enumerate(boundary_entity_map):
                # assign subdomain
                # 4 is for number of nodes in quad element
                # NOTE: we should find a different way to do this that doesn't assume quad elements if
                #    we plan to expand to other elements. -klb
                idx = int(np.where(cell_of_facet_mesh.array==xx)[0]/el_facets)
                boundary_subdomains.append(self.subdomains.values[idx])
                # assign orientation
                for j in range(3):
                    boundary_e1.x.array[3*i+j] = self.EE1.x.array[3*idx+j]
                    boundary_e2.x.array[3*i+j] = self.EE2.x.array[3*idx+j]
                    boundary_e3.x.array[3*i+j] = self.EE3.x.array[3*idx+j]

            boundary_frame = [boundary_e1, boundary_e2, boundary_e3]
            boundary_subdomains = np.array(boundary_subdomains, dtype=np.int32)
            boundary_num_cells = boundary_mesh.topology.index_map(boundary_mesh.topology.dim).size_local
            boundary_cells = np.arange(boundary_num_cells, dtype=np.int32)
            boundary_subdomains = dolfinx.mesh.meshtags(boundary_mesh, boundary_mesh.topology.dim, boundary_cells, boundary_subdomains)

            return boundary_subdomains, boundary_frame, boundary_facets
            # Mapping the orinetation data from quad mesh to boundary. The alternative is to use local_frame_1D(self.left_submesh["mesh"]).
            # Either of both can be used in local_boun subroutine 

        self.left_submesh["subdomains"], self.left_submesh["frame"], self.left_submesh["facets"] = _build_boundary_subdomains(self.left_submesh)
        self.right_submesh["subdomains"], self.right_submesh["frame"], self.right_submesh["facets"] = _build_boundary_subdomains(self.right_submesh)

       # def compute_boundary_stiffness_timo(self):
           #     left_stiffness=opensg.compute_solidtimo_boun(
           #         self.material_database[0],
           #         self.left_submesh)[1]

          #      right_stiffness=opensg.compute_solidtimo_boun(
          #          self.material_database[0],
          #          self.right_submesh)[1]    

      #          return  left_stiffness, right_stiffness
      #  self.left_stiffness, self.right_stiffness=compute_boundary_stiffness_timo(self)
        self.meshdata = {
            "mesh": self.mesh,
            "subdomains": self.subdomains,
            "frame": self.frame,
            }

        return


class SolidBladeMesh:
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
   #     self.sections = mesh_data["sections"]
        self.element_orientations = mesh_data["elementOrientations"]
        self._generate_layup_id()
        self._generate_material_database()

    def _generate_layup_id(self):
        layCt=-1
        self.mat_name=[]
        self.elLayID=np.zeros((self.num_elements))
        for es in self.sets['element']:
            if es['labels'][0] is not None:
                self.mat_name.append(es['name'])
                layCt += 1
                for eli in es['labels']:
                    self.elLayID[eli-1]=layCt
        return

    def _generate_material_database(self):
   #     material_database = dict()
        material_parameters,density=[],[]
        mat_names=[material['name'] for material in self.materials]

        for i, mat in enumerate(self.mat_name):
            es=self.materials[mat_names.index(mat)]
            material_parameters.append(np.array((np.array(es['E']),np.array(es['G']),es['nu'])).flatten())
            density.append(es['rho'])
          #  material_dict = dict()

          #  material_dict["id"] = i

           # elastic = material['elastic']
           # material_dict["E"] = elastic['E']
           # material_dict["G"] = elastic['G']
           # material_dict["nu"] = elastic['nu']
           # material_dict["rho"] = elastic['rho']
         #   material_database[material['name']] = material_dict
        #######
        ############            
        self.material_database = material_parameters,density
        return

    def generate_segment_mesh(self, segment_index, filename):

        file = open(filename,'w')

        file.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n')
       # newNumNds = np.max(segment_node_labels)

        file.write(str(self.num_nodes) + '\n')

        for i, nd in enumerate(self.nodes):
            nd=nd[0].split()
            ln = [str(i+1),str(nd[2]),str(nd[0]),str(nd[1])] # Making x-axis as beam axis
            file.write(' '.join(ln) + '\n')

        file.write('$EndNodes\n$Elements\n')

    #    newNumEls = np.max(segment_element_labels)
        file.write(str(self.num_elements) + '\n')

        for j,eli in enumerate(self.elements):

                ln = [str(j+1)]
                ln.append('5')
                ln.append('2')
                ln.append(str(1))
                ln.append(str(1))
                ell=eli[0].split()
                for n in ell:
                    ln.append(n)
                file.write(' '.join(ln) + '\n')

        file.write('$EndElements\n')
        file.close()

        self._generate_layup_id()
        self._generate_material_database()

        # initialize segmentmesh object
        segment_mesh = SolidBladeSegmentMesh(
        #    segment_node_labels=segment_node_labels,
         #   segment_element_labels=segment_element_labels,
         #   segment_element_layer_id=segment_element_layer_id,
            segment_index=segment_index,
        #    elLayID=self.elLayID,
            parent_blade_mesh=self,
         #   mat_param=self.material_database
            msh_file=filename)
        return segment_mesh


class ShellBladeSegmentMesh():
    """A class representing a segment of a wind turbine blade mesh.

    A segment is defined as the part of the blade between two fixed points along the blade span.
    Given a set of N span points along the blade, there are N-1 segments defined between each consecutive pair
    of span points. For example, the segment indexed by 0 is defined between the span points indexed by 0 and 1.

    This class manages the data and methods for analyzing the structural properties of a blade segment,
    including computing ABD matrices and stiffness properties.

    Attributes
    ----------
    mesh : dolfinx.mesh.Mesh
        The FEniCS/DOLFINx mesh object for this segment
    subdomains : dolfinx.mesh.MeshTags
        Tags identifying different regions/materials in the mesh
    left_submesh : dict
        Data for the left boundary submesh
    right_submesh : dict
        Data for the right boundary submesh
    layup_database : dict
        Database containing layup information including:
            - mat_names: list of material names
            - thick: list of layer thicknesses
            - angle: list of fiber angles
            - nlay: list of number of layers
    """
    def __init__(
        self,
        segment_node_labels,
        segment_element_labels,
        segment_element_layer_id,
        segment_index,
        layup_database,
        parent_blade_mesh,
        msh_file):
        """Initialize a SegmentMesh object.

        Parameters
        ----------
        segment_node_labels : array[int]
            Labels for nodes in this segment
        segment_element_labels : array[int]
            Labels for elements in this segment
        segment_element_layer_id : array[int]
            Layer IDs for each element
        segment_index : int
            Index of this segment in the blade
        layup_database : dict
            Database containing layup information
        parent_blade_mesh : BladeMesh
            Reference to the parent blade mesh object
        msh_file : str
            Path to the GMSH mesh file
        """
        self.segment_node_labels = segment_node_labels
        self.segment_element_labels = segment_element_labels
        self.segment_element_layer_id = segment_element_layer_id
        self.segment_index = segment_index
        self.layup_database = layup_database
        self.blade_mesh = parent_blade_mesh

        self.mesh, self.subdomains, self.boundaries = gmshio.read_from_msh(msh_file, MPI.COMM_WORLD,0, gdim=3)
        self.original_cell_index = self.mesh.topology.original_cell_index # Original cell Index from mesh file
        lnn = self.subdomains.values[:]-1
        self.num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        cells = np.arange(self.num_cells, dtype=np.int32)

        self.subdomains = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim, cells, np.array(lnn,dtype=np.int32))

        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1

        # self._generate_layup_data()
        self._build_local_orientations()
        self._build_boundary_submeshdata()

        return


    def _build_local_orientations(self):
        """Build local orientation vectors for each element.

        This method constructs the local coordinate system for each element
        based on the element orientations provided in the mesh data.

        Returns
        -------
        tuple
            Three dolfinx.fem.Function objects representing the local coordinate
            vectors (e1, e2, e3) for each element.
        """
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
        self.frame = frame
        return frame


    def _build_boundary_submeshdata(self):
        """Build submesh data for the left and right boundaries.

        This method extracts and processes the mesh data for the boundary
        regions of the segment, which are needed for applying boundary
        conditions and computing boundary stiffness properties.

        Returns
        -------
        tuple
            Two dictionaries containing the submesh data for the left and
            right boundaries respectively.
        """
        pp = self.mesh.geometry.x

        is_left_boundary, is_right_boundary = opensg.utils.shell.generate_boundary_markers(
            min(pp[:,0]), max(pp[:,0]))

        left_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_left_boundary)
        right_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_right_boundary)

        left_mesh, left_entity_map, left_vertex_map, left_geom_map = dolfinx.mesh.create_submesh(
            self.mesh, self.fdim, left_facets)
        right_mesh, right_entity_map, right_vertex_map, right_geom_map = dolfinx.mesh.create_submesh(
            self.mesh, self.fdim, right_facets)

        self.left_submesh = {
            "mesh": left_mesh,
            "entity_map": left_entity_map,
            "vertex_map": left_vertex_map,
            "geom_map": left_geom_map,
            "marker": is_left_boundary}

        self.right_submesh = {
            "mesh": right_mesh,
            "entity_map": right_entity_map,
            "vertex_map": right_vertex_map,
            "geom_map": right_geom_map,
            "marker": is_right_boundary}

        self.mesh.topology.create_connectivity(2,1)  # (quad mesh topology, boundary(1D) mesh topology)
        cell_of_facet_mesh = self.mesh.topology.connectivity(2,1)

        # NOTE: found a different way to determine connectivity that didn't need this block -klb
        # Cell to Edge connectivity
        # cell_edge_map = []
        # for i in range(self.num_cells):
        #     c = []
        #     for k in range(4): # 4 is used as number of edges in a quad element
        #         c.append((cell_of_facet_mesh.array[4*i+k])) 
        #     cell_edge_map.append(c)
        # cell_edge_map = np.ndarray.flatten(np.array(cell_edge_map))

        # generate subdomains
        def _build_boundary_subdomains(boundary_meshdata):
            boundary_mesh = boundary_meshdata["mesh"]
            boundary_entity_map = boundary_meshdata["entity_map"]
            boundary_marker = boundary_meshdata["marker"]
            boundary_VV = dolfinx.fem.functionspace(
                boundary_mesh, basix.ufl.element("DG", boundary_mesh.topology.cell_name(), 0, shape=(3, )))

            boundary_e1 = dolfinx.fem.Function(boundary_VV)
            boundary_e2 = dolfinx.fem.Function(boundary_VV)
            boundary_n = dolfinx.fem.Function(boundary_VV)

            boundary_facets = dolfinx.mesh.locate_entities(boundary_mesh, self.fdim, boundary_marker)

            # TODO: review the subdomain assingments with akshat
            boundary_subdomains = []
            for i, xx in enumerate(boundary_entity_map):
                # assign subdomain
                # 4 is for number of nodes in quad element
                # NOTE: we should find a different way to do this that doesn't assume quad elements if
                #    we plan to expand to other elements. -klb
                idx = int(np.where(cell_of_facet_mesh.array==xx)[0]/4)
                boundary_subdomains.append(self.subdomains.values[idx])
                # assign orientation
                for j in range(3):
                    boundary_e1.x.array[3*i+j] = self.EE1.x.array[3*idx+j]
                    boundary_e2.x.array[3*i+j] = self.EE2.x.array[3*idx+j]
                    boundary_n.x.array[3*i+j] = self.N.x.array[3*idx+j]

            boundary_frame = [boundary_e1, boundary_e2, boundary_n]
            boundary_subdomains = np.array(boundary_subdomains, dtype=np.int32)
            boundary_num_cells = boundary_mesh.topology.index_map(boundary_mesh.topology.dim).size_local
            boundary_cells = np.arange(boundary_num_cells, dtype=np.int32)
            boundary_subdomains = dolfinx.mesh.meshtags(boundary_mesh, boundary_mesh.topology.dim, boundary_cells, boundary_subdomains)

            return boundary_subdomains, boundary_frame, boundary_facets
            # Mapping the orinetation data from quad mesh to boundary. The alternative is to use local_frame_1D(self.left_submesh["mesh"]).
            # Either of both can be used in local_boun subroutine 

        self.left_submesh["subdomains"], self.left_submesh["frame"], self.left_submesh["facets"] = _build_boundary_subdomains(self.left_submesh)
        self.right_submesh["subdomains"], self.right_submesh["frame"], self.right_submesh["facets"] = _build_boundary_subdomains(self.right_submesh)

        return self.left_submesh, self.right_submesh

    def compute_ABD(self):
        """Compute the ABD (stiffness) matrices for the segment.

        This method computes the ABD matrices that relate forces and moments
        to strains and curvatures for each unique layup in the segment.

        Returns
        -------
        list
            List of 6x6 numpy arrays representing the ABD matrices for each
            unique layup in the segment.
        """
        nphases = max(self.subdomains.values[:]) + 1
        ABD_ = []
        for i in range(nphases):
            ABD_.append(opensg.compute_ABD_matrix(
                thick=self.layup_database["thick"][i],
                nlay=self.layup_database["nlay"][i],
                mat_names=self.layup_database["mat_names"][i],
                angle=self.layup_database["angle"][i],
                material_database=self.blade_mesh.material_database
                ))

        print('Computed',nphases,'ABD matrix')
        return ABD_


    # def plot(self):
    #     import pyvista
    #     pyvista.start_xvfb()
    #     u_topology, u_cell_types, u_geometry=dolfinx.plot.vtk_mesh(self.blade_mesh.mesh,self. mesh.blade_mesh.topology.dim)
    #     grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    #     grid.cell_data["Marker"] = self.blade_mesh.subdomains.values[:]
    #     grid.set_active_scalars("Marker")
    #     u_plotter = pyvista.Plotter()
    #     u_plotter.add_mesh(grid)
    #     u_plotter.show_axes()
    #     #u_plotter.view_xy() # z is beam axis
    #     u_plotter.show()

    def compute_stiffness_EB(self, ABD):
        """Compute the Euler-Bernoulli beam stiffness matrix.

        Parameters
        ----------
        ABD : list
            List of ABD matrices for each unique layup

        Returns
        -------
        numpy.ndarray
            4x4 stiffness matrix for Euler-Bernoulli beam theory
        """
        # extract object data
        mesh = self.mesh
        frame = self.frame
        subdomains = self.subdomains

        D_eff = opensg.compute_stiffness_EB_blade_segment(
            ABD,
            mesh,
            frame,
            subdomains,
            self.left_submesh,
            self.right_submesh)

        return D_eff

    def compute_boundary_stiffness_timo(self, ABD):
        """Compute the Timoshenko beam stiffness matrices for the boundaries.

        Parameters
        ----------
        ABD : list
            List of ABD matrices for each unique layup

        Returns
        -------
        tuple
            Left and right boundary 6x6 Timoshenko stiffness matrices
        """
        left_stiffness = core.compute_timo_boun(ABD, self.left_submesh)[1]

        right_stiffness = core.compute_timo_boun(
            ABD,
            self.right_submesh["mesh"],
            self.right_submesh["subdomains"],
            self.right_submesh["frame"],
            self.nullspace, # quad nullspace
            self.right_submesh["nullspace"],
            self.nphases)[1]

        return left_stiffness, right_stiffness

    def compute_stiffness(self, ABD):
        """Compute all stiffness matrices for the segment.

        This method computes both the Euler-Bernoulli and Timoshenko
        stiffness matrices for the segment and its boundaries.

        Parameters
        ----------
        ABD : list
            List of ABD matrices for each unique layup

        Returns
        -------
        tuple
            Contains:
            - segment_timo_stiffness: 6x6 Timoshenko stiffness matrix
            - segment_eb_stiffness: 4x4 Euler-Bernoulli stiffness matrix
            - l_timo_stiffness: 6x6 left boundary Timoshenko stiffness matrix
            - r_timo_stiffness: 6x6 right boundary Timoshenko stiffness matrix
        """
        return core.compute_stiffness(
            ABD=ABD,
            mesh=self.mesh,
            subdomains=self.subdomains,
            l_submesh=self.left_submesh,
            r_submesh=self.right_submesh
        )

