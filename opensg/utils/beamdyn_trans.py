# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 21:39:47 2025

@author: bagla0
"""
import numpy as np


def write_beamdyn_prop(folder, wt_name, radial_stations, beam_stiff, beam_inertia, mu):
    n_pts = len(radial_stations)

        
    propFileName= 'bd_props_'+wt_name + '.inp'
    
    
    file = open(folder +'/'+propFileName, 'w')
    file.write(' ------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------\n')
    file.write(' Test Format 1\n')
    file.write(' ---------------------- BLADE PARAMETERS --------------------------------------\n')
    file.write('%u   station_total    - Number of blade input stations (-)\n' % (n_pts))
    file.write(' 1   damp_type        - Damping type: 0: no damping; 1: damped\n')
    file.write('  ---------------------- DAMPING COEFFICIENT------------------------------------\n')
    file.write('   mu1        mu2        mu3        mu4        mu5        mu6\n')
    file.write('   (-)        (-)        (-)        (-)        (-)        (-)\n')
    file.write('\t %.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e\n' % (mu[0], mu[1], mu[2], mu[3], mu[4], mu[5])) 
    file.write(' ---------------------- DISTRIBUTED PROPERTIES---------------------------------\n')
    
    for i in range(n_pts):
        file.write('\t %.6f \n' % (radial_stations[i]))
        # write stiffness matrices
        for j in range(6):
            file.write('\t %.16e \t %.16e \t %.16e \t %.16e \t %.16e \t %.16e\n' % (
            beam_stiff[i, j, 0], beam_stiff[i, j, 1], beam_stiff[i, j, 2], beam_stiff[i, j, 3], beam_stiff[i, j, 4],
            beam_stiff[i, j, 5]))
        file.write('\n')

        # write inertia properties
        for j in range(6):
            file.write('\t %.16e \t %.16e \t %.16e \t %.16e \t %.16e \t %.16e\n' % (
            beam_inertia[i, j, 0], beam_inertia[i, j, 1], beam_inertia[i, j, 2], beam_inertia[i, j, 3],
            beam_inertia[i, j, 4], beam_inertia[i, j, 5]))
        file.write('\n')
        # ToDO: check correct translation of stiffness and mass matrices from VABS and anbax !!!
    file.close()

    print('Finished writing BeamDyn_Blade File')

    return propFileName

def transformMatrixToBeamDyn(beam_stiff,beam_inertia):
  #  beamDynData={}

    B = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])  # NEW transformation matrix
    T = np.dot(np.identity(3), np.linalg.inv(B))
    
    nStations, _,_=np.shape(beam_stiff)

    for i_station in range(nStations):
        beam_stiff[i_station,:,:]=trsf_sixbysix(beam_stiff[i_station,:,:], T)
        beam_inertia[i_station,:,:]=trsf_sixbysix(beam_inertia[i_station,:,:], T)
   
    return(beam_stiff,beam_inertia)

def trsf_sixbysix(M, T):
    """
    Transform six-by-six compliance/stiffness matrix. 
    change of reference frame in engineering (or Voigt) notation.
    
    Parameters
    ----------
    M : np.ndarray
        6x6 Siffness or Mass Matrix
    T : np.ndarray
        Transformation Matrix
        
    Returns
    ----------
    res : np.ndarray
        Transformed 6x6 matrix
    """

    TS_1 = np.dot(np.dot(T.T, M[0:3, 0:3]), T)
    TS_2 = np.dot(np.dot(T.T, M[3:6, 0:3]), T)
    TS_3 = np.dot(np.dot(T.T, M[0:3, 3:6]), T)
    TS_4 = np.dot(np.dot(T.T, M[3:6, 3:6]), T)

    tmp_1 = np.vstack((TS_1, TS_2))
    tmp_2 = np.vstack((TS_3, TS_4))
    res = np.hstack((tmp_1, tmp_2))
    return res

def transform_beam_matrices(beam_stiff, beam_inertia):
    """
    Transform 6x6 beam stiffness and inertia matrices to a new reference frame.

    Parameters:
    ----------
    beam_stiff : np.ndarray
        6x6 stiffness matrix
    beam_inertia : np.ndarray
        6x6 inertia matrix

    Returns:
    ----------
    beam_stiff_tr : np.ndarray
        Transformed 6x6 stiffness matrix
    beam_inertia_tr : np.ndarray
        Transformed 6x6 inertia matrix
    """
    # Example transformation matrix (customize as needed)
    B = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]) 
    T = np.dot(np.identity(3), np.linalg.inv(B)) 


    beam_stiff_tr = trsf_sixbysix(beam_stiff, T)
    beam_inertia_tr = trsf_sixbysix(beam_inertia, T)
    return beam_stiff_tr, beam_inertia_tr

def beam_reaction(file_name):
    """Parse beam reaction forces from output file.

    Reads and parses beam reaction forces from a simulation output file,
    extracting force and moment components for each blade segment.

    Parameters
    ----------
    file_name : str
        Base name of the output file (without .out extension)

    Returns
    -------
    list
        List of beam force data for each segment, where each segment contains
        6 components (3 forces, 3 moments) with labels and values.
    """
    data = np.loadtxt(file_name + ".out", delimiter=",", skiprows=0, dtype=str)
    index = data[1].split()
    last_data = data[-1].split()
    pp = 7
    # beam_f=[[[index[pp+k],float(last_data[pp+k])] for k in range(6)]] ;if root also needed
    beam_force, beam_disp = [], []
    num_segment = int((len(index) - 13) / 15)
    pp = 13

    for seg in range(num_segment):
        beam_seg_reac = []
        for f in range(6):
            sc = pp + 30 * (f) + seg
            if f > 2:
                sc = pp + num_segment * (3 + f) + seg
            beam_seg_reac.append([index[sc], float(last_data[sc])])
        #  print(f,index[sc])
        beam_force.append(beam_seg_reac)
    
        beam_seg_disp=[]
        for f in range(9,15):
            sc=pp+num_segment*(f)+seg
            beam_seg_disp.append([index[sc],float(last_data[sc])])
        beam_disp.append(beam_seg_disp)
    return beam_force, beam_disp

def tension_center(flex):
    ff=flex[2,2]*flex[3,3]-flex[2,3]*flex[2,3]
    return [(flex[2,2]*flex[0,3]-flex[2,3]*flex[0,2])/ff, (flex[2,3]*flex[0,3]-flex[3,3]*flex[0,2])/ff]