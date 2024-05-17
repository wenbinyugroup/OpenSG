# updated

import numpy as np
import yaml
from yaml import CLoader as cLd

## Define input parameters

meshYaml = 'bar_urc_shell_mesh.yaml'  ## the name of the yaml file containing the whole blade mesh
secInd = np.linspace(1,2,2) ## the index of the spanwise section you want
mshFile = 'bar_urc_shell.msh'
oriFile = 'shell.orientation'
layupFile = 'shell.layup'
matFile = 'shell.material.properties'

## Read the mesh yaml file

inFile = open(meshYaml,'r')
meshData = yaml.load(inFile,Loader=cLd)
inFile.close()

## Extract the mesh for the section
nodes = meshData['nodes']
numNds = len(nodes)
elements = meshData['elements']
numEls = len(elements)

ndNewLabs = -1*np.ones(numNds,dtype=int)
elNewLabs = -1*np.ones(numEls,dtype=int)
elLayID = -1*np.ones(numEls,dtype=int)

layCt = 0
for es in meshData['sets']['element']:
    nmLst = es['name'].split('_')
    if(len(nmLst) > 2):
        if(int(nmLst[1]) in secInd):
            for eli in es['labels']:
                elNewLabs[eli] = 1
                elLayID[eli] = layCt
                for nd in elements[eli]:
                    if(nd > -1):
                        ndNewLabs[nd] = 1
            layCt += 1

eLab = 1
for i, e in enumerate(elNewLabs):
    if(e == 1):
        elNewLabs[i] = eLab
        eLab += 1

ndLab = 1
for i, n in enumerate(ndNewLabs):
    if(n == 1):
        ndNewLabs[i] = ndLab
        ndLab += 1

## Write material properties file
outFile = open(matFile,'w')
outFile.write('material_parameters=[]\n\n')
matDic = dict()
for i, m in enumerate(meshData['materials']):
    matDic[m['name']] = i
    el = m['elastic']
    eProps = el['E']
    eProps.extend(el['G'])
    eProps.extend(el['nu'])
    ln = 'material_parameters.append(' + str(eProps) + ')\n\n'
    outFile.write(ln)
    
outFile.close()

## Write layups file
outFile = open(layupFile,'w')
layCt = 0
for sec in meshData['sections']:
    nmLst = sec['elementSet'].split('_')
    if(len(nmLst) > 2):
        if(int(nmLst[1]) in secInd):
            layup = sec['layup']
            nlst = str(len(layup))
            lcst = str(layCt)
            lnLst = [lcst,nlst]
            for l in layup:
                lnLst.append(str(matDic[l[0]]))
            outFile.write(' '.join(lnLst) + '\n')
            lnLst = [lcst,nlst]
            for l in layup:
                lnLst.append(str(l[1]))
            outFile.write(' '.join(lnLst) + '\n')
            lnLst = [lcst,nlst]
            for l in layup:
                lnLst.append(str(l[2]))
            outFile.write(' '.join(lnLst) + '\n')
            layCt += 1
        
outFile.close()

## Write element orientations file
outFile = open(oriFile,'w')
for i, eo in enumerate(meshData['elementOrientations']):
    elab = elNewLabs[i]
    if(elNewLabs[i] > -1):
        ln = [str(elab)]
        for j in range(0,9):
            ln.append(str(eo[j]))
        outFile.write(' '.join(ln) + '\n')

outFile.close()

## Write .msh file

outFile = open(mshFile,'w')

outFile.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n')
newNumNds = np.max(ndNewLabs)
outFile.write(str(newNumNds) + '\n')

for i, nd in enumerate(nodes):
    lab = ndNewLabs[i]
    if(lab > -1):
        ln = [str(lab),str(nd[2]),str(nd[0]),str(nd[1])]
        outFile.write(' '.join(ln) + '\n')

outFile.write('$EndNodes\n$Elements\n')

newNumEls = np.max(elNewLabs)
outFile.write(str(newNumEls) + '\n')

for i, el in enumerate(elements):
    lab = elNewLabs[i]
    if(lab > -1):
        ln = [str(lab)]
        if(el[3] == -1):
            ln.append('2')
        else:
            ln.append('3')
        ln.append('2')
        ln.append('2')
        ln.append(str(elLayID[i]))
        for nd in el:
            if(nd > -1):
                ln.append(str(ndNewLabs[nd]))
        outFile.write(' '.join(ln) + '\n')
outFile.write('$EndElements\n')

outFile.close()
