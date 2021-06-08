function [crop,r1,r2,c1,c2]=autocrop(im)

im2=rgb2gray(im);
im2(:,:)=255-im2(:,:);
r1=1;c1=1;
for row=1:size(im,1)
    rowsum=0;
    for col=1:size(im,2)
        rowsum=rowsum+double(im2(row,col));
    end
    if rowsum>0
        r1=row;
        break;
    end
end
for row=r1+10:size(im,1)
    rowsum=0;
    for col=1:size(im,2)
        rowsum=rowsum+double(im2(row,col));
    end
    if (rowsum==0)
        r2=row;  
        break;
    end
end
for col=1:size(im,2)
    colsum=0;
    for row=1:size(im,1)
        colsum=colsum+double(im2(row,col));
    end
    if colsum>0
        c1=col;
        break;
    end
end
for col=c1+10:size(im,2)
    colsum=0;
    for row=1:size(im,1)
        colsum=colsum+double(im2(row,col));
    end
    if colsum==0
        c2=col;
        break;
    end
end
crop=uint8(255*ones(r2-r1+1,c2-c1+1,3));
crop(:,:,1:3)=im(r1:r2,c1:c2,1:3);
end