"""Input/output utilities for OpenSG.

This module provides functions for reading and writing mesh data in various formats,
including YAML and GMSH formats.
"""

import yaml
import numpy as np


def load_yaml(yaml_file):
    """Load mesh data from a YAML file.
    
    Parameters
    ----------
    yaml_file : str
        Path to the YAML file containing mesh data
        
    Returns
    -------
    dict
        Dictionary containing the mesh data with keys:
        - nodes: list of node coordinates
        - elements: list of element definitions
        - sets: dictionary of element and node sets
        - materials: dictionary of material definitions
        - sections: dictionary of section definitions
        - elementOrientations: list of element orientation matrices
    """
    with open(yaml_file, 'r') as file:
        mesh_data = yaml.load(file, Loader=yaml.CLoader)
    return mesh_data

def write_yaml(data, yaml_file):
    """Write data to a YAML file.
    
    Parameters
    ----------
    data : dict
        Data to write to the file
    yaml_file : str
        Path to the output YAML file
        
    Returns
    -------
    None
    """
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, Loader=yaml.CLoader)
    return

# TODO write a function to validate mesh data schema
def validate_mesh_data(mesh_data):
    """Validate the structure and content of mesh data.
    
    This function is not yet implemented. It will check that the mesh data
    contains all required fields and that they have the correct structure.
    
    Parameters
    ----------
    mesh_data : dict
        Dictionary containing mesh data to validate
        
    Returns
    -------
    bool
        True if the data is valid, False otherwise
        
    Raises
    ------
    NotImplementedError
        This function is not yet implemented
    """
    raise NotImplementedError("Mesh data validation is not yet implemented")

def write_mesh(filename, blade_mesh):
    """Write mesh data to a GMSH format file.
    
    Parameters
    ----------
    filename : str
        Path to the output mesh file
    blade_mesh : BladeMesh
        BladeMesh object containing the mesh data to write
        
    Returns
    -------
    None
    
    Notes
    -----
    The mesh is written in GMSH 2.2 format with the following sections:
    - MeshFormat: version and type information
    - Nodes: node coordinates
    - Elements: element definitions with tags
    """
    mesh_file = open(filename, 'w')

    mesh_file.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n')
    newNumNds = np.max(ndNewLabs)
    mesh_file.write(str(newNumNds) + '\n')

    for i, nd in enumerate(nodes):
        lab = ndNewLabs[i]
        if(lab > -1):
            ln = [str(lab),str(nd[2]),str(nd[0]),str(nd[1])]
        #  ln = [str(lab),str(nd[0]),str(nd[1]),str(nd[2])]
            mesh_file.write(' '.join(ln) + '\n')

    mesh_file.write('$EndNodes\n$Elements\n')

    newNumEls = np.max(elNewLabs)
    mesh_file.write(str(newNumEls) + '\n')

    for i, el in enumerate(elements):
        lab = elNewLabs[i]
        if(lab > -1):
            ln = [str(lab)]
            if(el[3] == -1):
                ln.append('2')
            else:
                ln.append('3')
            ln.append('2')
            ln.append(str(elLayID[i]+1))
            ln.append(str(elLayID[i]+1))
            for nd in el:
                if(nd > -1):
                    ln.append(str(ndNewLabs[nd]))
            mesh_file.write(' '.join(ln) + '\n')
    mesh_file.write('$EndElements\n')

    mesh_file.close()