import yaml
import numpy as np


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        mesh_data = yaml.load(file, Loader=yaml.CLoader)
    return mesh_data

def write_yaml(data, yaml_file):
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, Loader=yaml.CLoader)
    return

# TODO write a function to validate mesh data schema


def write_mesh(filename, blade_mesh):
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