0 0 0 0             # analysis elem_flag trans_flag temp_flag
1 4 3 3 1  0         # nSG nnode nelem nmate nslave

1  -1     	    # nodal coordinates: node_no y1 y2 y3
2   0.  
3   0.5 
4   1 

1 1 1 2 0  0 0	    # element material type & connectivity: element_no mtype node1 node2 
2 2 2 3 0 0 0
3 3 3 4 0 0 0

 1 4               # slave_node master_node

1 1 1               # mtype isotropy ntemp
100 0.5             # temperature density   
50e9 15.2e9 15.2e9  
4.7e9  4.7e9 3.28e9
0.254 0.254 0.428


2 0 1            
100 0.6
2600000 0.


3 2 1
100 0.6
0.26000000E+07   0.00000000E+00   0.00000000E+00   0.00000000E+00   0.00000000E+00   0.00000000E+00
		 0.26000000E+07   0.00000000E+00   0.00000000E+00   0.00000000E+00   0.00000000E+00
				  0.26000000E+07   0.00000000E+00   0.00000000E+00   0.00000000E+00
  					           0.13000000E+07   0.00000000E+00   0.00000000E+00
								    0.13000000E+07   0.00000000E+00
										     0.13000000E+07
2.0  # volume of SG

