import numpy as np
import dolfinx
import basix
from dolfinx.io import gmshio
from mpi4py import MPI
import opensg
import gmsh

from opensg.mesh.solidsegment import SolidSegmentMesh


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
        segment_mesh = SolidSegmentMesh(
        #    segment_node_labels=segment_node_labels,
         #   segment_element_labels=segment_element_labels,
         #   segment_element_layer_id=segment_element_layer_id,
            segment_index=segment_index, 
        #    elLayID=self.elLayID,
            parent_blade_mesh=self, 
         #   mat_param=self.material_database
            msh_file=filename)
        return segment_mesh

