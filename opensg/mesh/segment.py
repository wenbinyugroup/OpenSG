from opensg.core import shell as core

import opensg
import basix
import dolfinx
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI



class SegmentMesh():
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

    # def _generate_layup_data(self):
    #     layup_database = dict()

    #     mat_names, thick, angle, nlay = [], [], [], []
    #     for section in self.blade_mesh.sections:
    #         name_components = section['elementSet'].split('_')
    #         if(len(name_components) > 2):
    #             material_name, t, an = [], [], []
    #             if(int(name_components[1]) == self.segment_index):
    #                 layup = section['layup'] # layup = [[material_name: str, thickness: float, angle:]]
    #                 nlay.append(len(layup))
    #                 for layer in layup:
    #                     material_name.append(layer[0])     
    #                 mat_names.append(material_name)
    #                 for layer in layup:
    #                     t.append(layer[1])
    #                 thick.append(t)
    #                 for layer in layup:
    #                     an.append(layer[2])
    #                 angle.append(an) 


    #     layup_database["mat_names"] = mat_names
    #     layup_database["thick"] = thick
    #     layup_database["angle"] = angle
    #     layup_database["nlay"] = nlay
    #     self.layup_database = layup_database

    #     return layup_database


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