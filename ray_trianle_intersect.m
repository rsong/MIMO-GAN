function z_value=ray_trianle_intersect(p,pa,pb,pc)

a=pa(1:2);
b=pb(1:2);
c=pc(1:2);

SameSide1=same_side(p,a,b,c);
SameSide2=same_side(p,b,a,c);
SameSide3=same_side(p,c,a,b);

if not(SameSide1 & SameSide2 & SameSide3)
    z_value=[];
    return;
end;

x1=a(1);
x2=b(1);
x3=c(1);
x4=p(1);
y1=a(2);
y2=b(2);
y3=c(2);
y4=p(2);

x_temp=det([det([x1 y1;x2 y2]) x1-x2;det([x3 y3;x4 y4]) x3-x4])/det([x1-x2 y1-y2;x3-x4 y3-y4]);
y_temp=det([det([x1 y1;x2 y2]) y1-y2;det([x3 y3;x4 y4]) y3-y4])/det([x1-x2 y1-y2;x3-x4 y3-y4]);
    
p_temp=[x_temp y_temp];

dist1=norm(a-p_temp);
dist2=norm(b-p_temp);


z_temp=(dist2*pa(3)+dist1*pb(3))/(dist1+dist2);

dist1=norm(c-p);
dist2=norm(p_temp-p);

z_value=(dist2*pc(3)+dist1*z_temp)/(dist1+dist2);
