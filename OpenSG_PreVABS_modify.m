clear all
%% User-Input (PreVABS .msh file reqired)

fname='SG40_cut_1111';  % file name
data=readlines(strcat(fname,'.msh'));
%% Nodes
 if data(4)=="$Nodes"
     disp('Working with Coordinate data')
 else
     disp('ensure $Nodes on Line 4 in msh file')  
 end
 nodes=str2num(split(data(5)));
 for j=1:nodes
        k=5+j;
        n1=split(data(k));
        data(k)=strcat(n1(1),{'   '}, n1(3),{'   '}, n1(4),{'   '}, n1(2));
 end
%% Elements
 if data(k+2)=="$Elements"
          disp('Working with material ID')
 else
     sprintf('ensure $Elements on Line %d in msh file',k+2)
 end
 elements=str2num(split(data(k+3)));

 for j=1:elements
    kk=k+j+3;
    n1=split(data(kk));
    n1(4)=num2str(str2num(n1(4))-1);
    data(kk)=join(n1);
 end 
 %% Writing to .msh file
fname=strcat('m',fname);
fileID = fopen(strcat(fname,'.msh'),'w');
for i =1:length(data)
    fprintf(fileID,'%s',data(i));
    fprintf(fileID,'\n');
end
fclose(fileID);


