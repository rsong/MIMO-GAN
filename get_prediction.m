load('mimogan.mat');
for i=1:size(results_sa,1)
    image1=results_sa(i,:,:,:);
    image1_sa=reshape(image1,[224,224,3]);
    image1_sa_norm = image1_sa;
    image1_sa_norm(image1_sa<0)=0;
    image1_sa_norm = (image1_sa_norm-min(image1_sa_norm(:)))./(max(image1_sa_norm(:))-min(image1_sa_norm(:)));
    image1_output=rgb2gray(image1_sa_norm);
    image_output=uint8(image1_output*255);
    imwrite(image_output,strcat('.\prediction\',num2str(i),'.jpg'),'jpg');
end