import numpy as np
import dolfinx
import basix
from dolfinx.io import gmshio
from mpi4py import MPI
from ufl import dot, as_tensor, rhs
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import form, petsc, Function, locate_dofs_topological, apply_lifting, set_bc
from scipy.sparse import csr_matrix
import scipy
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
        segment_mesh = SegmentMesh(
            segment_node_labels=segment_node_labels,
            segment_element_labels=segment_element_labels,
            segment_element_layer_id=segment_element_layer_id,
            segment_index=segment_index,
            layup_database=layup_database,
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
        layup_database,
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
        pp = self.mesh.geometry.x

        is_left_boundary, is_right_boundary = opensg.generate_boundary_markers(
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
    
    def compute_stiffness_EB_boundary(self, ABD):
        m_l = opensg.compute_eb_blade_segment_boundary(ABD, self.left_submesh)
        m_r = opensg.compute_eb_blade_segment_boundary(ABD, self.right_submesh)
        
        return m_l, m_r
    
    def compute_boundary_stiffness_timo(self, ABD):

        left_stiffness = opensg.compute_timo_boun(ABD, self.left_submesh)[1]
        
        right_stiffness = opensg.compute_timo_boun(
            ABD, 
            self.right_submesh["mesh"],
            self.right_submesh["subdomains"],
            self.right_submesh["frame"],
            self.nullspace, # quad nullspace
            self.right_submesh["nullspace"],
            self.nphases)[1]
        
        return left_stiffness, right_stiffness
    
    def compute_stiffness(self, ABD):
        return opensg.compute_stiffness(
            ABD=ABD,
            mesh=self.mesh,
            subdomains=self.subdomains,
            l_submesh=self.left_submesh,
            r_submesh=self.right_submesh
        )
        

"""
    def compute_stiffness_EB(self, ABD):
        # extract object data
        mesh = self.mesh
        frame = self.frame
        subdomains = self.subdomains
        tdim=mesh.topology.dim
        fdim = tdim - 1
        nphases = max(self.subdomains.values[:]) + 1
        
        pp = mesh.geometry.x # point data
        x_min, x_max=min(pp[:,0]), max(pp[:,0])
        # Initialize terms
        e_l, V_l, dvl, v_l, x_l, dx_l = opensg.local_boun(
            self.left_submesh["mesh"], self.left_submesh["frame"] ,self.left_submesh["subdomains"])
        
        e_r, V_r, dvr, v_r, x_r, dx_r = opensg.local_boun(
            self.right_submesh["mesh"], self.right_submesh["frame"] ,self.right_submesh["subdomains"])
        
        self.left_submesh["nullspace"] = opensg.compute_nullspace(V_l)
        self.right_submesh["nullspace"] = opensg.compute_nullspace(V_r)
        
        A_l = opensg.A_mat(ABD, e_l,x_l,dx_l,self.left_submesh["nullspace"],v_l,dvl, nphases)
        A_r = opensg.A_mat(ABD, e_r,x_r,dx_r,self.right_submesh["nullspace"],v_r,dvr, nphases)
        
        V0_l = opensg.solve_boun(ABD, self.left_submesh, nphases)
        
        V0_r = opensg.solve_boun(ABD, self.right_submesh, nphases)
        # The local_frame_l(self.left_submesh["mesh"]) can be replaced with frame_l, if we want to use mapped orientation from given direction cosine matrix (orien mesh data-yaml)

        # Quad mesh
        e, V, dv, v_, x, dx = opensg.local_boun(mesh, frame, subdomains)
        V0, Dle, Dhe, Dhd, Dld, D_ed, D_dd, D_ee, V1s = opensg.initialize_array(V)
        mesh.topology.create_connectivity(1, 2)
        self.left_submesh["mesh"].topology.create_connectivity(1, 1)
        self.right_submesh["mesh"].topology.create_connectivity(1, 1)

        # Obtaining coefficient matrix AA and BB with and without bc applied.
        # Note: bc is applied at boundary dofs. We define v2a containing all dofs of entire wind blade.
        boundary_dofs = locate_dofs_topological(V, fdim, np.concatenate((self.right_submesh["entity_map"], self.left_submesh["entity_map"]), axis=0))
        F2=sum([dot(dot(as_tensor(ABD[i]), opensg.gamma_h(e,x,dv)), opensg.gamma_h(e,x,v_))*dx(i) for i in range(nphases)])  
        v2a=Function(V) # b default, v2a has zero value for all. 
        bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs) # This shows only boundary_dofs are taken for v2a under bc, which are zero (known) as input.
        a = form(F2)
        
        B = assemble_matrix(a)   # Obtain coefficient matrix without BC applied: BB
        B.assemble()
        ai, aj, av=B.getValuesCSR()
        BB = csr_matrix((av, aj, ai))
        BB = BB.toarray()  
        
        A = assemble_matrix(a,[bc])  # Obtain coefficient matrix with BC applied: AA
        A.assemble()
        ai, aj, av=A.getValuesCSR()
        AA = csr_matrix((av, aj, ai))
        AA=AA.toarray()
        avg=np.trace(AA)/len(AA)     

        # averaging is done so that all terms are of same order. Note after appliying bc at [A=assemble_matrix(a,[bc])], the dofs of
        # coefficientmatrix has only 1 replaced at that dofs. 
        for i,xx in enumerate(av):
            if xx==1:
                av[i]=avg        
                                
        AA_csr = csr_matrix((av, aj, ai))
        AAA = AA_csr.toarray() 
        AA = scipy.sparse.csr_matrix(AAA) 

        # Assembly
        # Running for 4 different F vector. However, F has bc applied to it where, stored known values of v2a is provided for each loop (from boun solve).
        for p in range(4): # 4 load cases meaning 
            # Boundary 
            v2a = Function(V)
            v2a = opensg.dof_mapping_quad(V, v2a,V_l,V0_l[:,p], self.left_submesh["facets"], self.left_submesh["entity_map"]) 
            v2a = opensg.dof_mapping_quad(V, v2a,V_r,V0_r[:,p], self.right_submesh["facets"], self.right_submesh["entity_map"])  
            
            # quad mesh
            F2=sum([dot(dot(as_tensor(ABD[i]),opensg.gamma_e(e,x)[:,p]), opensg.gamma_h(e,x,v_))*dx(i) for i in range(nphases)])  
            bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
            F = petsc.assemble_vector(form(rhs(F2)))
            apply_lifting(F, [a], [[bc]]) # apply bc to rhs vector (Dhe)
            set_bc(F, [bc])
            with F.localForm() as local_F:
                for i in boundary_dofs:
                    for k in range(3):
                        # F[3*i+k]=avg*F[3*i+k] # normalize small terms
                        local_index = 3 * i + k
                        local_F[local_index] = avg * local_F[local_index]
                    
            V0[:,p]=  scipy.sparse.linalg.spsolve(AA, F, permc_spec=None, use_umfpack=True) # obtain sol: E* V1s = b*
            Dhe[:,p]= scipy.sparse.csr_array(BB).dot(V0[:,p])
            
        D1 = np.matmul(V0.T,-Dhe) 
        for s in range(4):
            for k in range(4): 
                f = dolfinx.fem.form(sum([dot(dot(opensg.gamma_e(e,x).T,as_tensor(ABD[i])),opensg.gamma_e(e,x))[s,k]*dx(i) for i in range(nphases)]))
                D_ee[s,k]=dolfinx.fem.assemble_scalar(f)
        L = (x_max - x_min)
        D_eff= D_ee + D1
        D_eff=D_eff/L  # L is divided because of 3D shell mesh and corresponding beam length need to divided.
        #--------------------------------------Printing Output Data---------------------------------------
        print('  ')  
        print('Stiffness Matrix')
        np.set_printoptions(precision=4)
        print(np.around(D_eff))
        
        return D_eff
"""
        

class MeshData:
    def __init__(self, mesh):
        # What data is required to start?
        
        pass
    
    def add_submesh_attributes(self,
        entity_map,
        vertex_map,
        geom_map,
        submesh_facets):
        pass
    
    def process_data(self):
        pass
        # what downstream data can be generated with the starting data?
        
    def process_data_with_args(self, args):
        pass
        # what data needs external information to be generated?
        # how do we track that this data is not soley derived from the init data?
        
            # Mapping the orinetation data from quad mesh to boundary. The alternative is to use local_frame_1D(self.left_submesh["mesh"]).
            # Either of both can be used in local_boun subroutine 
        
            # self.left_submesh["subdomains"], self.left_submesh["frame"], self.left_submesh["facets"] = subdomains_boundary(
            #     self.left_submesh["mesh"], is_left_boundary, self.left_submesh["entity_map"]) 
            # self.right_submesh["subdomains"], self.right_submesh["frame"], self.right_submesh["facets"] = subdomains_boundary(
            #     self.right_submesh["mesh"], is_right_boundary, self.right_submesh["entity_map"])
            
            
def _generate_submesh_subdomains(submesh_data, mesh_data):
        mesh_data.mesh.topology.create_connectivity(2,1)  # (quad mesh topology, boundary(1D) mesh topology)
        cell_of_facet_mesh = mesh_data.mesh.topology.connectivity(2,1)
        submesh_VV = dolfinx.fem.functionspace(
            submesh_data.mesh, basix.ufl.element("DG", submesh_data.mesh.topology.cell_name(), 0, shape=(3, )))
        
        submesh_e1 = dolfinx.fem.Function(submesh_VV)
        submesh_e2 = dolfinx.fem.Function(submesh_VV)
        submesh_n = dolfinx.fem.Function(submesh_VV)
        
        submesh_facets = dolfinx.mesh.locate_entities(submesh_data.mesh, self.fdim, submesh_marker)
        submesh_subdomains = []
        for i, xx in enumerate(submesh_data.entity_map):
            # assign subdomain
            idx = int(np.where(cell_of_facet_mesh.array==xx)[0]/4) # 4 is for number of nodes in quad element
            submesh_subdomains.append(mesh_data.subdomains.values[idx])
            # assign orientation
            for j in range(3):
                submesh_e1.x.array[3*i+j] = mesh_data.EE1.x.array[3*idx+j]
                submesh_e2.x.array[3*i+j] = mesh_data.EE2.x.array[3*idx+j]
                submesh_n.x.array[3*i+j] = mesh_data.N.x.array[3*idx+j]

        submesh_frame = [submesh_e1, submesh_e2, submesh_n]
        submesh_subdomains = np.array(submesh_subdomains, dtype=np.int32)
        submesh_num_cells = submesh_data.mesh.topology.index_map(submesh_data.mesh.topology.dim).size_local 
        submesh_cells = np.arange(submesh_num_cells, dtype=np.int32)
        submesh_subdomains = dolfinx.mesh.meshtags(
            submesh_data.mesh, submesh_data.mesh.topology.dim, submesh_cells, submesh_subdomains)
        
        return submesh_subdomains, submesh_frame, submesh_facets