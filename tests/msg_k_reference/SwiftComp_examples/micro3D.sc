0 0 0  0        # analysis elem_flag trans_flag temp_flag
3 32 2 2 0 0	# nSG nnode nelem nmate nslave

1  1 -1 -1    	# nodal coordinates: node_no y1 y2 y3
2  1 1  -1  
3  1 1  0   
4  1 -1  0  
5  -1 -1 -1    	 
6  -1 1  -1 
7  -1 1  0  
8  -1 -1  0 
9  1 -1 1     	 
10  1 1  1  
11  -1 1  1   
12  -1 -1  1  
13 1 0 -1 
14 0 1 -1 
15 -1 0 -1
16 0 -1 -1
17 1 0 0 
18 0 1 0 
19 -1 0 0
20 0 -1 0
21 1 -1  -0.5
22 1 1  -0.5 
23 -1 1 -0.5 
24 -1 -1 -0.5
25 1 0 1 
26 0 1 1 
27 -1 0 1
28 0 -1 1
29 1 -1  0.5
30 1 1  0.5 
31 -1 1 0.5 
32 -1 -1 0.5
 

1 1 1 2 6 5 4 3  7  8 13 14 15 16 17 18 19 20 21 22 23 24   # element material type & connectivity: element_no mtype node1 node2 
2 2 4 3 7 8 9 10 11 12 17 18 19 20 25 26 27 28 29 30 31 32

 

1 1 1            # mtype isotropy ntemp
100 0.5          # temperature density   
50e9 15.2e9 15.2e9
4.7e9  4.7e9 3.28e9
0.254 0.254 0.428


2 0 1
100 0.6
2600000 0.

8           # volume of SG
