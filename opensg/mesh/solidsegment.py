import numpy as np
import dolfinx
import basix
from dolfinx.io import gmshio
from mpi4py import MPI
import opensg
import gmsh
import yaml
import tempfile
import os


class SolidSegmentMesh():
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


class StandaloneSolidSegmentMesh:
    """A standalone class representing a segment of a wind turbine blade solid mesh.
    
    This class is designed to work independently without requiring a parent SolidBladeMesh.
    It reads all necessary data directly from segment YAML files and provides
    the same computational capabilities as the original SolidSegmentMesh.
    
    This class combines the functionality of SolidBladeMesh and SolidSegmentMesh
    to process segment data in the same way as the original workflow.
    
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
    material_database : tuple
        Tuple containing (material_parameters, density) for solid analysis
    elLayID : numpy.ndarray
        Array mapping elements to their layup IDs
    mat_name : list
        List of material names
    """
    
    def __init__(self, segment_yaml_file):
        """Initialize a StandaloneSolidSegmentMesh object from a YAML file.
        
        Parameters
        ----------
        segment_yaml_file : str
            Path to the segment YAML file containing all necessary data
        """
        
        # Load segment data from YAML
        with open(segment_yaml_file, 'r') as f:
            segment_data = yaml.safe_load(f)
        
        # Extract data from YAML (same as SolidBladeMesh.__init__)
        self.nodes = segment_data['nodes']
        self.num_nodes = len(self.nodes)
        
        self.elements = segment_data['elements']
        self.num_elements = len(self.elements)
        
        self.sets = segment_data["sets"]
        self.materials = segment_data["materials"]        
        self.element_orientations = segment_data["elementOrientations"]
        
        # Generate layup ID and material database (same as SolidBladeMesh)
        self._generate_layup_id()
        self._generate_material_database()
        
        # Build mesh (same as SolidBladeMesh.generate_segment_mesh + SolidSegmentMesh.__init__)
        self._build_mesh()
        
        # Build local orientations and boundary data (same as SolidSegmentMesh)
        self._build_local_orientations()
        self._build_boundary_submeshes()
        
    def _generate_layup_id(self):
        """Generate layup ID mapping (same as SolidBladeMesh._generate_layup_id)."""
        lay_ct = -1
        self.mat_name = []
        self.elLayID = np.zeros((self.num_elements))
        
        for es in self.sets['element']:
            if es['labels'][0] is not None:
                self.mat_name.append(es['name'])
                lay_ct += 1
                for eli in es['labels']:
                    self.elLayID[eli-1] = lay_ct
        return
        
    def _generate_material_database(self):
        """Generate material database (same as SolidBladeMesh._generate_material_database)."""
        material_parameters, density = [], []
        mat_names = [material['name'] for material in self.materials]
        
        for i, mat in enumerate(self.mat_name):
            es = self.materials[mat_names.index(mat)]
            material_parameters.append(np.array((np.array(es['E']), np.array(es['G']), es['nu'])).flatten())
            density.append(es['rho'])
            
        self.material_database = (material_parameters, density)
        return
        
    def _build_mesh(self):
        """Build the mesh from the segment data (same as SolidBladeMesh.generate_segment_mesh + SolidSegmentMesh.__init__)."""
        # Create temporary MSH file for DOLFINx compatibility (same as SolidBladeMesh.generate_segment_mesh)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.msh', delete=False) as temp_msh:
            msh_filename = temp_msh.name
            
            # Write GMSH format
            temp_msh.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n')
            temp_msh.write(str(self.num_nodes) + '\n')
            
            for i, nd in enumerate(self.nodes):
                # Handle node format (same as SolidBladeMesh.generate_segment_mesh)
                nd = nd[0].split()
                ln = [str(i+1), str(nd[2]), str(nd[0]), str(nd[1])]  # Making x-axis as beam axis
                temp_msh.write(' '.join(ln) + '\n')
            
            temp_msh.write('$EndNodes\n$Elements\n')
            temp_msh.write(str(self.num_elements) + '\n')
            
            for j, eli in enumerate(self.elements):
                ln = [str(j+1)]
                ln.append('5')
                ln.append('2')
                ln.append(str(1))   
                ln.append(str(1))    
                ell = eli[0].split()
                for n in ell:
                    ln.append(n)
                temp_msh.write(' '.join(ln) + '\n')
            
            temp_msh.write('$EndElements\n')
        
        # Load mesh using DOLFINx (same as SolidSegmentMesh.__init__)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        self.mesh, self.subdomains, self.boundaries = gmshio.read_from_msh(msh_filename, MPI.COMM_WORLD, 0, gdim=3)
        
        self.original_cell_index = self.mesh.topology.original_cell_index
        
        # Create layup ID mapping (same as SolidSegmentMesh.__init__)
        # For standalone, the segment YAML already contains only segment elements
        # So the layup IDs we generated are already the segment layup IDs
        # We just need to map them to the DOLFINx mesh cell order using original_cell_index
        
        lnn = []
        for k in self.original_cell_index:
            lnn.append(self.elLayID[k])
        
        self.num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        cells = np.arange(self.num_cells, dtype=np.int32)
        
        self.subdomains = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim, cells, np.array(lnn, dtype=np.int32))
        
        # Store elLayID for compatibility with SolidSegmentMesh
        # This should be the layup IDs in the order of the DOLFINx mesh cells
        # self.elLayID = np.array(lnn, dtype=np.float64)  # Match the dtype from SolidBladeMesh
        
        # Set up topology
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
        
        # Clean up temporary file
        os.unlink(msh_filename)
    
    def _build_local_orientations(self):
        """Build local orientation vectors for each element (same as SolidSegmentMesh._build_local_orientations)."""
        # Local Orientation (DG0 function) of solid mesh element
        VV = dolfinx.fem.functionspace(
            self.mesh, basix.ufl.element(
            "DG", self.mesh.topology.cell_name(),
            0, shape=(3, )))
        EE1, EE2, EE3 = dolfinx.fem.Function(VV), dolfinx.fem.Function(VV), dolfinx.fem.Function(VV)

        # Process orientations (same as SolidSegmentMesh._build_local_orientations)
        orientations = []
        for i, eo in enumerate(self.element_orientations):
            o = []
            for k in range(9):
                o.append(eo[k])
            orientations.append(o)

        # Store orientation for each element using original_cell_index mapping
        for k, ii in enumerate(self.original_cell_index):
            # Store data to DG0 functions 
            EE2.x.array[3*k], EE2.x.array[3*k+1], EE2.x.array[3*k+2] = orientations[ii][5], orientations[ii][3], orientations[ii][4]  # e2
            EE3.x.array[3*k], EE3.x.array[3*k+1], EE3.x.array[3*k+2] = orientations[ii][8], orientations[ii][6], orientations[ii][7]   # e3 
            EE1.x.array[3*k], EE1.x.array[3*k+1], EE1.x.array[3*k+2] = orientations[ii][2], orientations[ii][0], orientations[ii][1]  # e1
        
        self.EE1 = EE1
        self.EE2 = EE2
        self.EE3 = EE3
        self.frame = [EE1, EE2, EE3]
    
    def _build_boundary_submeshes(self):
        """Build submesh data for the left and right boundaries (same as SolidSegmentMesh._build_boundary_submeshes)."""
        pp = self.mesh.geometry.x
        x_min, x_max = min(pp[:,0]), max(pp[:,0])
        mean = 0.5*(x_min + x_max)  # Mid origin for taper segments
        
        blade_length = x_max - x_min
        
        self.left_origin, self.right_origin, self.taper_origin = [], [], []
        self.left_origin.append(float(x_min)/blade_length)
        self.right_origin.append(float(x_max)/blade_length)
        self.taper_origin.append(float(mean)/blade_length)
        
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

        self.right_submesh = {
            "mesh": right_mesh,
            "entity_map": right_entity_map,
            "vertex_map": right_vertex_map,
            "geom_map": right_geom_map,
            "marker": is_right_boundary}

        self.mesh.topology.create_connectivity(3, 2)
        cell_of_facet_mesh = self.mesh.topology.connectivity(3, 2)

        # Generate subdomains for boundaries
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

            boundary_subdomains = []
            el_facets = 6  # For solid elements (hexahedra)
            for i, xx in enumerate(boundary_entity_map):
                # assign subdomain
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

        self.left_submesh["subdomains"], self.left_submesh["frame"], self.left_submesh["facets"] = _build_boundary_subdomains(self.left_submesh)
        self.right_submesh["subdomains"], self.right_submesh["frame"], self.right_submesh["facets"] = _build_boundary_subdomains(self.right_submesh)
        
        # Create meshdata dictionary for compatibility
        self.meshdata = {
            "mesh": self.mesh,
            "subdomains": self.subdomains,
            "frame": self.frame,
        }