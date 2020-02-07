function out=ZigzagMtx2Vector(in)
 
% out=ZigzagMtx2Vector(in)
% convert zigzag order matrix to a vector
% e.g. 
%      1 2 6 
%      3 5 7 
%      4 8 9
%  ==>
%      1 2 3 4 5 6 7 8 9
%
% Author : Guan-Ming Su
% Date : 10/08/00

Dmtx=size(in,1);
out=zeros(1,Dmtx*Dmtx);
TurnFlg=0;   % 0 for down trun, 1 for up trun
index=2;
out(1)=in(1,1);

for w=3:Dmtx+1
  switch TurnFlg
    case 0,
      for i=1:w-1
        j=w-i;
        out(index)=in(i,j);
        index=index+1;        
      end
      TurnFlg=1;
    case 1,
      for i=w-1:-1:1
        j=w-i;
        out(index)=in(i,j);        
        index=index+1;
      end
      TurnFlg=0;        
   end
end

if mod(Dmtx,2)==0
   TurnFlg=1;
else
   TurnFlg=0;
end      

for w=Dmtx+2:2*Dmtx
  switch TurnFlg
    case 0,
      for i=w-Dmtx:Dmtx
        j=w-i;
        out(index)=in(i,j);        
        index=index+1;        
      end
      TurnFlg=1;
    case 1,
      for i=Dmtx:-1:w-Dmtx
        j=w-i;
        out(index)=in(i,j);        
        index=index+1;
      end
      TurnFlg=0;        
   end
end

