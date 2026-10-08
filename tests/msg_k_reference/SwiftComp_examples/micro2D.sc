0 0 0 0          # analysis elem_flag trans_flag temp_flag
2 13 2 2  0 0     # nSG nnode nelem nmate nslave

1  -1 -1    	  # nodal coordinates: node_no y2 y3
2  1 -1  
3  1 0   
4  1  1  
5  -1 1 
6  -1 0 
7  0 -1 
8 0 0   
9 0 1   
10 1 -0.5 
11  1 0.5 
12  -1 0.5
13  -1 -0.5 
 

1 1 1 2 3 6 7  10 8 13 0  	 # element material type & connectivity: element_no mtype node1 node2 
2 2 6 3 4 5 8 11 9 12 0
 

1 1 1             # mtype isotropy ntemp
100 0.5           # temperature density   
50e9 15.2e9 15.2e9
4.7e9  4.7e9 3.28e9
0.254 0.254 0.428


2 0 1           
100 0.6
2600000 0.

4                # volume of SG
