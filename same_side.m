function S=same_side(p1,p2,a,b)
p1=[p1 0];
p2=[p2 0];
a=[a 0];
b=[b 0];
cp1=cross(b-a,p1-a);
cp2=cross(b-a,p2-a);
if dot(cp1,cp2)>=0;
    S=1;
else
    S=0;
end;