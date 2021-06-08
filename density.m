function W = density(face,vertex)
A=triangulation2adjacency(face);
if size(vertex,1)<size(vertex,2)
    vertex = vertex';
end

vring=zeros(length(vertex),1);



for m = 1:length(vertex)
    vring(m)=sum(A(m,:)~=0);
end
[i,j,s] = find(sparse(A));
d = sum( (vertex(i,:) - vertex(j,:)).^2, 2);
ww = sparse(i,j,double(d));
W=sum(ww)./vring';
W=full(W);
% W=1./(W+eps);
%  W=W.^2;
W=(W-min(W))./(max(W)-min(W));
end

