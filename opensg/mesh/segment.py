import gmsh
import opensg
from opensg.core import shell as core


import basix
import dolfinx
import numpy as np
import yaml
from dolfinx.io import gmshio
from mpi4py import MPI


import os
import tempfile


class ShellSegmentMesh:
    """A standalone class representing a segment of a wind turbine blade mesh.

    This class is designed to work independently without requiring a parent BladeMesh.
    It reads all necessary data directly from segment YAML files and provides
    the same computational capabilities as the original SegmentMesh.

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
    material_database : dict
        Processed database of material properties
    """

    def __init__(self, segment_yaml_file):
        """Initialize a StandaloneSegmentMesh object from a YAML file.

        Parameters
        ----------
        segment_yaml_file : str
            Path to the segment YAML file containing all necessary data
        """

        # Load segment data from YAML
        with open(segment_yaml_file, "r") as f:
            segment_data = yaml.safe_load(f)

        # Extract data from YAML
        self.nodes = segment_data["nodes"]
        self.elements = segment_data["elements"]
        self.sets = segment_data["sets"]
        self.materials = segment_data["materials"]
        self.sections = segment_data["sections"]
        self.element_orientations = segment_data["elementOrientations"]

        # Build mesh
        self._build_mesh()

        # Build layup database
        self._build_layup_database()

        # Build local orientations and boundary data
        self._build_local_orientations()
        self._build_boundary_submeshdata()

    def _build_layup_database(self):
        """Build the layup database from the segment data.

        This method creates a dictionary containing the layup information
        for each section in the segment.
        """
        # # Generate material database
        self.material_database = dict()
        for i, material in enumerate(self.materials):
            material_dict = dict()
            material_dict["id"] = i
            elastic = material["elastic"]
            material_dict["E"] = elastic["E"]
            material_dict["G"] = elastic["G"]
            material_dict["nu"] = elastic["nu"]
            self.material_database[material["name"]] = material_dict

        # Create layup database from sections
        mat_names, thick, angle, nlay = [], [], [], []
        for section in self.sections:
            layup = section["layup"]
            nlay.append(len(layup))
            m, t, an = [], [], []
            for layer in layup:
                m.append(layer[0])
                t.append(layer[1])
                an.append(layer[2])
            mat_names.append(m)
            thick.append(t)
            angle.append(an)

        self.layup_database = {
            "mat_names": mat_names,
            "thick": thick,
            "angle": angle,
            "nlay": nlay,
        }

    def generate_mesh_file(self, filename):
        """Generate a GMSH format mesh file from the segment data.

        Parameters
        ----------
        filename : str
            Path where the mesh file should be written
        """
        with open(filename, "w") as msh_file:
            # Write GMSH format
            msh_file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
            msh_file.write(f"{len(self.nodes)}\n")

            for i, node in enumerate(self.nodes):
                msh_file.write(f"{i + 1} {node[2]} {node[0]} {node[1]}\n")

            msh_file.write("$EndNodes\n$Elements\n")
            msh_file.write(f"{len(self.elements)}\n")

            # Create subdomain mapping from sets
            subdomains = np.zeros(len(self.elements), dtype=int)
            for i, element_set in enumerate(self.sets["element"]):
                for element_idx in element_set["labels"]:
                    subdomains[element_idx] = i

            for i, element in enumerate(self.elements):
                element_type = "3" if len(element) == 4 else "2"  # 3=quad, 2=tri
                msh_file.write(
                    f"{i + 1} {element_type} 2 {subdomains[i] + 1} {subdomains[i] + 1}"
                )
                for node_idx in element:
                    msh_file.write(f" {node_idx + 1}")
                msh_file.write("\n")

            msh_file.write("$EndElements\n")

        return

    def _build_mesh(self):
        """Build the mesh from the segment data.

        This method creates a DOLFINx mesh object from the segment data.
        """
        # Create temporary MSH file for DOLFINx compatibility
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".msh", delete=False
        ) as temp_msh:
            msh_filename = temp_msh.name

        # Generate the mesh file content
        self.generate_mesh_file(msh_filename)

        # Load mesh using DOLFINx
        self.mesh, self.subdomains, self.boundaries = gmshio.read_from_msh(
            msh_filename, MPI.COMM_WORLD, 0, gdim=3
        )

        self.original_cell_index = (
            self.mesh.topology.original_cell_index
        )  # Original cell Index from mesh file
        lnn = self.subdomains.values[:] - 1
        self.num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        cells = np.arange(self.num_cells, dtype=np.int32)

        self.subdomains = dolfinx.mesh.meshtags(
            self.mesh, self.mesh.topology.dim, cells, np.array(lnn, dtype=np.int32)
        )

        # Clean up temporary file
        os.unlink(msh_filename)

        # Set up topology
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1

        return

    def _build_local_orientations(self):
        """Build local orientation vectors for each element.

        This method constructs the local coordinate system for each element
        based on the element orientations provided in the mesh data.
        """
        # Local Orientation (DG0 function) of quad mesh element
        VV = dolfinx.fem.functionspace(
            self.mesh,
            basix.ufl.element("DG", self.mesh.topology.cell_name(), 0, shape=(3,)),
        )
        EE1, EE2, N = (
            dolfinx.fem.Function(VV),
            dolfinx.fem.Function(VV),
            dolfinx.fem.Function(VV),
        )

        # Store orientation for each element using original_cell_index mapping
        # This ensures orientations are mapped to the correct mesh cells
        for k, ii in enumerate(self.original_cell_index):
            orientation = self.element_orientations[ii]
            # Store data to DG0 functions
            EE2.x.array[3 * k], EE2.x.array[3 * k + 1], EE2.x.array[3 * k + 2] = (
                orientation[5],
                orientation[3],
                orientation[4],
            )  # e2
            N.x.array[3 * k], N.x.array[3 * k + 1], N.x.array[3 * k + 2] = (
                orientation[8],
                orientation[6],
                orientation[7],
            )  # e3
            EE1.x.array[3 * k], EE1.x.array[3 * k + 1], EE1.x.array[3 * k + 2] = (
                orientation[2],
                orientation[0],
                orientation[1],
            )  # e1

        self.EE1 = EE1
        self.N = N
        self.EE2 = EE2
        self.frame = [EE1, EE2, N]

    def _build_boundary_submeshdata(self):
        """Build submesh data for the left and right boundaries.

        This method extracts and processes the mesh data for the boundary
        regions of the segment, which are needed for applying boundary
        conditions and computing boundary stiffness properties.
        """
        pp = self.mesh.geometry.x

        is_left_boundary, is_right_boundary = (
            opensg.utils.shell.generate_boundary_markers(min(pp[:, 0]), max(pp[:, 0]))
        )

        left_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_left_boundary
        )
        right_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_right_boundary
        )

        left_mesh, left_entity_map, left_vertex_map, left_geom_map = (
            dolfinx.mesh.create_submesh(self.mesh, self.fdim, left_facets)
        )
        right_mesh, right_entity_map, right_vertex_map, right_geom_map = (
            dolfinx.mesh.create_submesh(self.mesh, self.fdim, right_facets)
        )

        self.left_submesh = {
            "mesh": left_mesh,
            "entity_map": left_entity_map,
            "vertex_map": left_vertex_map,
            "geom_map": left_geom_map,
            "marker": is_left_boundary,
        }

        self.right_submesh = {
            "mesh": right_mesh,
            "entity_map": right_entity_map,
            "vertex_map": right_vertex_map,
            "geom_map": right_geom_map,
            "marker": is_right_boundary,
        }

        self.mesh.topology.create_connectivity(2, 1)
        cell_of_facet_mesh = self.mesh.topology.connectivity(2, 1)

        # Generate subdomains for boundaries
        def _build_boundary_subdomains(boundary_meshdata):
            boundary_mesh = boundary_meshdata["mesh"]
            boundary_entity_map = boundary_meshdata["entity_map"]
            boundary_marker = boundary_meshdata["marker"]
            boundary_VV = dolfinx.fem.functionspace(
                boundary_mesh,
                basix.ufl.element(
                    "DG", boundary_mesh.topology.cell_name(), 0, shape=(3,)
                ),
            )

            boundary_e1 = dolfinx.fem.Function(boundary_VV)
            boundary_e2 = dolfinx.fem.Function(boundary_VV)
            boundary_n = dolfinx.fem.Function(boundary_VV)

            boundary_facets = dolfinx.mesh.locate_entities(
                boundary_mesh, self.fdim, boundary_marker
            )

            boundary_subdomains = []
            for i, xx in enumerate(boundary_entity_map):
                # assign subdomain
                idx = int(np.where(cell_of_facet_mesh.array == xx)[0] / 4)
                boundary_subdomains.append(self.subdomains.values[idx])
                # assign orientation
                for j in range(3):
                    boundary_e1.x.array[3 * i + j] = self.EE1.x.array[3 * idx + j]
                    boundary_e2.x.array[3 * i + j] = self.EE2.x.array[3 * idx + j]
                    boundary_n.x.array[3 * i + j] = self.N.x.array[3 * idx + j]

            boundary_frame = [boundary_e1, boundary_e2, boundary_n]
            boundary_subdomains = np.array(boundary_subdomains, dtype=np.int32)
            boundary_num_cells = boundary_mesh.topology.index_map(
                boundary_mesh.topology.dim
            ).size_local
            boundary_cells = np.arange(boundary_num_cells, dtype=np.int32)
            boundary_subdomains = dolfinx.mesh.meshtags(
                boundary_mesh,
                boundary_mesh.topology.dim,
                boundary_cells,
                boundary_subdomains,
            )

            return boundary_subdomains, boundary_frame, boundary_facets

        (
            self.left_submesh["subdomains"],
            self.left_submesh["frame"],
            self.left_submesh["facets"],
        ) = _build_boundary_subdomains(self.left_submesh)
        (
            self.right_submesh["subdomains"],
            self.right_submesh["frame"],
            self.right_submesh["facets"],
        ) = _build_boundary_subdomains(self.right_submesh)

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
            ABD_.append(
                core.compute_ABD_matrix(
                    thick=self.layup_database["thick"][i],
                    nlay=self.layup_database["nlay"][i],
                    mat_names=self.layup_database["mat_names"][i],
                    angle=self.layup_database["angle"][i],
                    material_database=self.material_database,
                )
            )

        print("Computed", nphases, "ABD matrix")
        return ABD_

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
            r_submesh=self.right_submesh,
        )


class SolidSegmentMesh:
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
        with open(segment_yaml_file, "r") as f:
            segment_data = yaml.safe_load(f)

        # Extract data from YAML (same as SolidBladeMesh.__init__)
        self.nodes = segment_data["nodes"]
        self.num_nodes = len(self.nodes)

        self.elements = segment_data["elements"]
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

        for es in self.sets["element"]:
            if es["labels"][0] is not None:
                self.mat_name.append(es["name"])
                lay_ct += 1
                for eli in es["labels"]:
                    self.elLayID[eli - 1] = lay_ct
        return

    def _generate_material_database(self):
        """Generate material database (same as SolidBladeMesh._generate_material_database)."""
        material_parameters, density = [], []
        mat_names = [material["name"] for material in self.materials]

        for i, mat in enumerate(self.mat_name):
            es = self.materials[mat_names.index(mat)]
            material_parameters.append(
                np.array((np.array(es["E"]), np.array(es["G"]), es["nu"])).flatten()
            )
            density.append(es["rho"])

        self.material_database = (material_parameters, density)
        return

    def generate_mesh_file(self, filename):
        """Generate a GMSH format mesh file from the solid segment data.

        Parameters
        ----------
        filename : str
            Path where the mesh file should be written
        """
        with open(filename, "w") as msh_file:
            # Write GMSH format
            msh_file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
            msh_file.write(str(self.num_nodes) + "\n")

            for i, nd in enumerate(self.nodes):
                # Handle node format (same as SolidBladeMesh.generate_segment_mesh)
                nd = nd[0].split()
                ln = [
                    str(i + 1),
                    str(nd[2]),
                    str(nd[0]),
                    str(nd[1]),
                ]  # Making x-axis as beam axis
                msh_file.write(" ".join(ln) + "\n")

            msh_file.write("$EndNodes\n$Elements\n")
            msh_file.write(str(self.num_elements) + "\n")

            for j, eli in enumerate(self.elements):
                ln = [str(j + 1)]
                ln.append("5")  # Element type 5 for solid elements
                ln.append("2")
                ln.append(str(1))
                ln.append(str(1))
                ell = eli[0].split()
                for n in ell:
                    ln.append(n)
                msh_file.write(" ".join(ln) + "\n")

            msh_file.write("$EndElements\n")

        return

    def _build_mesh(self):
        """Build the mesh from the segment data (same as SolidBladeMesh.generate_segment_mesh + SolidSegmentMesh.__init__)."""
        # Create temporary MSH file for DOLFINx compatibility (same as SolidBladeMesh.generate_segment_mesh)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".msh", delete=False
        ) as temp_msh:
            msh_filename = temp_msh.name

        # Generate the mesh file content
        self.generate_mesh_file(msh_filename)

        # Load mesh using DOLFINx (same as SolidSegmentMesh.__init__)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        self.mesh, self.subdomains, self.boundaries = gmshio.read_from_msh(
            msh_filename, MPI.COMM_WORLD, 0, gdim=3
        )

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

        self.subdomains = dolfinx.mesh.meshtags(
            self.mesh, self.mesh.topology.dim, cells, np.array(lnn, dtype=np.int32)
        )

        # Store elLayID for compatibility with SolidSegmentMesh
        # This should be the layup IDs in the order of the DOLFINx mesh cells
        # self.elLayID = np.array(lnn, dtype=np.float64)  # Match the dtype from SolidBladeMesh

        # Set up topology
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1

        # Clean up temporary file
        os.unlink(msh_filename)

        return

    def _build_local_orientations(self):
        """Build local orientation vectors for each element (same as SolidSegmentMesh._build_local_orientations)."""
        # Local Orientation (DG0 function) of solid mesh element
        VV = dolfinx.fem.functionspace(
            self.mesh,
            basix.ufl.element("DG", self.mesh.topology.cell_name(), 0, shape=(3,)),
        )
        EE1, EE2, EE3 = (
            dolfinx.fem.Function(VV),
            dolfinx.fem.Function(VV),
            dolfinx.fem.Function(VV),
        )

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
            EE2.x.array[3 * k], EE2.x.array[3 * k + 1], EE2.x.array[3 * k + 2] = (
                orientations[ii][5],
                orientations[ii][3],
                orientations[ii][4],
            )  # e2
            EE3.x.array[3 * k], EE3.x.array[3 * k + 1], EE3.x.array[3 * k + 2] = (
                orientations[ii][8],
                orientations[ii][6],
                orientations[ii][7],
            )  # e3
            EE1.x.array[3 * k], EE1.x.array[3 * k + 1], EE1.x.array[3 * k + 2] = (
                orientations[ii][2],
                orientations[ii][0],
                orientations[ii][1],
            )  # e1

        self.EE1 = EE1
        self.EE2 = EE2
        self.EE3 = EE3
        self.frame = [EE1, EE2, EE3]

    def _build_boundary_submeshes(self):
        """Build submesh data for the left and right boundaries (same as SolidSegmentMesh._build_boundary_submeshes)."""
        pp = self.mesh.geometry.x
        x_min, x_max = min(pp[:, 0]), max(pp[:, 0])
        mean = 0.5 * (x_min + x_max)  # Mid origin for taper segments

        blade_length = x_max - x_min

        self.left_origin, self.right_origin, self.taper_origin = [], [], []
        self.left_origin.append(float(x_min) / blade_length)
        self.right_origin.append(float(x_max) / blade_length)
        self.taper_origin.append(float(mean) / blade_length)

        is_left_boundary, is_right_boundary = (
            opensg.utils.solid.generate_boundary_markers(min(pp[:, 0]), max(pp[:, 0]))
        )

        left_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_left_boundary
        )
        right_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=self.fdim, marker=is_right_boundary
        )

        left_mesh, left_entity_map, left_vertex_map, left_geom_map = (
            dolfinx.mesh.create_submesh(self.mesh, self.fdim, left_facets)
        )
        right_mesh, right_entity_map, right_vertex_map, right_geom_map = (
            dolfinx.mesh.create_submesh(self.mesh, self.fdim, right_facets)
        )

        self.left_submesh = {
            "mesh": left_mesh,
            "entity_map": left_entity_map,
            "vertex_map": left_vertex_map,
            "geom_map": left_geom_map,
            "marker": is_left_boundary,
        }

        self.right_submesh = {
            "mesh": right_mesh,
            "entity_map": right_entity_map,
            "vertex_map": right_vertex_map,
            "geom_map": right_geom_map,
            "marker": is_right_boundary,
        }

        self.mesh.topology.create_connectivity(3, 2)
        cell_of_facet_mesh = self.mesh.topology.connectivity(3, 2)

        # Generate subdomains for boundaries
        def _build_boundary_subdomains(boundary_meshdata):
            boundary_mesh = boundary_meshdata["mesh"]
            boundary_entity_map = boundary_meshdata["entity_map"]
            boundary_marker = boundary_meshdata["marker"]
            boundary_VV = dolfinx.fem.functionspace(
                boundary_mesh,
                basix.ufl.element(
                    "DG", boundary_mesh.topology.cell_name(), 0, shape=(3,)
                ),
            )

            boundary_e1 = dolfinx.fem.Function(boundary_VV)
            boundary_e2 = dolfinx.fem.Function(boundary_VV)
            boundary_e3 = dolfinx.fem.Function(boundary_VV)

            boundary_facets = dolfinx.mesh.locate_entities(
                boundary_mesh, self.fdim, boundary_marker
            )

            boundary_subdomains = []
            el_facets = 6  # For solid elements (hexahedra)
            for i, xx in enumerate(boundary_entity_map):
                # assign subdomain
                idx = int(np.where(cell_of_facet_mesh.array == xx)[0] / el_facets)
                boundary_subdomains.append(self.subdomains.values[idx])
                # assign orientation
                for j in range(3):
                    boundary_e1.x.array[3 * i + j] = self.EE1.x.array[3 * idx + j]
                    boundary_e2.x.array[3 * i + j] = self.EE2.x.array[3 * idx + j]
                    boundary_e3.x.array[3 * i + j] = self.EE3.x.array[3 * idx + j]

            boundary_frame = [boundary_e1, boundary_e2, boundary_e3]
            boundary_subdomains = np.array(boundary_subdomains, dtype=np.int32)
            boundary_num_cells = boundary_mesh.topology.index_map(
                boundary_mesh.topology.dim
            ).size_local
            boundary_cells = np.arange(boundary_num_cells, dtype=np.int32)
            boundary_subdomains = dolfinx.mesh.meshtags(
                boundary_mesh,
                boundary_mesh.topology.dim,
                boundary_cells,
                boundary_subdomains,
            )

            return boundary_subdomains, boundary_frame, boundary_facets

        (
            self.left_submesh["subdomains"],
            self.left_submesh["frame"],
            self.left_submesh["facets"],
        ) = _build_boundary_subdomains(self.left_submesh)
        (
            self.right_submesh["subdomains"],
            self.right_submesh["frame"],
            self.right_submesh["facets"],
        ) = _build_boundary_subdomains(self.right_submesh)

        # Create meshdata dictionary for compatibility
        self.meshdata = {
            "mesh": self.mesh,
            "subdomains": self.subdomains,
            "frame": self.frame,
        }
