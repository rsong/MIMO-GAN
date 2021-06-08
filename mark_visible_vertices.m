function visibility_v = mark_visible_vertices(V,F,view_point)

% figure(1)
% trisurf(F,V(:,1),V(:,2),V(:,3),1,'edgealpha',0);
% daspect([1 1 1]);

if isequal(view_point,[0 0 -1]); 
    
    V(:,2:3)=-V(:,2:3);
    
else if not(isequal(view_point,[0 0 1]))

        view_point=view_point(:);
v_p=view_point/norm(view_point);
z_a=v_p;
x_a=cross(z_a,[0 ; 0;  1]);
x_a=x_a/norm(x_a);
y_a=cross(z_a,x_a);
y_a=y_a/norm(y_a);


Trans_Matrix=[x_a y_a z_a];
%V=inv(Trans_Matrix)*V';
V=Trans_Matrix\V';
V=V';

end;
end;

% figure(2)
% trisurf(F,V(:,1),V(:,2),V(:,3),1,'edgealpha',0);
% daspect([1 1 1]);

num_vertices=size(V,1);

Xv=V(:,1);
Yv=V(:,2);

Tx3=[Xv(F(:,1)) Xv(F(:,2)) Xv(F(:,3))];
Ty3=[Yv(F(:,1)) Yv(F(:,2)) Yv(F(:,3))];

maxTx = max(Tx3,[],2);
minTx = min(Tx3,[],2);

maxTy = max(Ty3,[],2);
minTy = min(Ty3,[],2);

visibility_v = ones(num_vertices,1);

for v=1:num_vertices;
    % disp(v);
    
    xx = V(v,1);
    yy = V(v,2);
    
    z_init = V(v,3);
    
    I_tri = find( (maxTx >= xx) .* (minTx <= xx) .* (maxTy >= yy) .* (minTy <= yy) );

    
    for ti=1:length(I_tri);
        t = I_tri(ti);
        X1=V(F(t,1),1);
        X2=V(F(t,2),1);
        X3=V(F(t,3),1);
        Y1=V(F(t,1),2);
        Y2=V(F(t,2),2);
        Y3=V(F(t,3),2);
        Z1=V(F(t,1),3);
        Z2=V(F(t,2),3);
        Z3=V(F(t,3),3);
        
        pa=[X1 Y1 Z1];
        pb=[X2 Y2 Z2];
        pc=[X3 Y3 Z3];
        
        p=[xx yy];
        
        if isequal(p,pa(1:2));
            z_value=pa(3);
        else if isequal(p,pb(1:2));
                z_value=pb(3);
            else if isequal(p,pc(1:2));
                    z_value=pc(3);
                else
                    z_value=ray_trianle_intersect([xx yy],pa,pb,pc);
                end;
            end;
        end;
        
        
        
       if z_value > z_init;
           visibility_v(v)=0;
           break;
       end;
    end;
    
end;




           