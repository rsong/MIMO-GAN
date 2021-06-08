
function [cor_score,p_score,auc_score] = sa_trans_eval(shapefolder, imagefolder, saliencyfolder, results_sa)

% 2D-to-3D saliency transfer and evaluation using the 3DVA dataset
% shapefolder: the path of the folder where the test 3D shapes are
% imagefolder: the path of the folder where the 2D views of the test 3D shapes are
% saliencyfolder: the path of the folder where the output saliency maps are saved
% cor_score: the LCC scores
% p_score: the p values
% auc_score: the auc scores

% This function works with the variable 'results_sa' containing the 2D saliency maps producded by the MIMO-GAN.

% Copyright (c) 2020 Ran Song

 % load results_sa

outputSize = 224;
nViews = 3; % The 3DVA dataset provides the fixation maps at 3 views for each 3D object.
shapedir = dir(shapefolder);
imagedir = dir(imagefolder);

num_shape=numel(shapedir);
cor_score = zeros(num_shape-2,3);
auc_score = zeros(num_shape-2,3);
p_score = zeros(num_shape-2,3);

for i=3:num_shape

    mesh = loadMesh( [shapedir(i).folder '\' shapedir(i).name] );
    viewpoints = load(['.\3DVA\test_viewpoints\' shapedir(i).name(1:end-4) '.txt']);
    
    % Normalise the mesh
    xn1=max(mesh.V(1,:));
    xn2=min(mesh.V(1,:));
    yn1=max(mesh.V(2,:));
    yn2=min(mesh.V(2,:));
    zn1=max(mesh.V(3,:));
    zn2=min(mesh.V(3,:));
    bbox=sqrt((xn1-xn2).^2+(yn1-yn2).^2+(zn1-zn2).^2);
    
    mesh.V(1,:)=double(mesh.V(1,:)-0.5*(xn1+xn2));
    mesh.V(2,:)=double(mesh.V(2,:)-0.5*(yn1+yn2));
    mesh.V(3,:)=double(mesh.V(3,:)-0.5*(zn1+zn2));
    mesh.F=double(mesh.F);
    
    % For efficiency, we operate on a simplified mesh
    if length(mesh.F)>20000
        [p,t] = perform_mesh_simplification(mesh.V',mesh.F',20000);
        knng=knnsearch(p,mesh.V');
    else
        p=mesh.V';
        t=mesh.F';
        knng=1:length(mesh.V);
    end
    
    W=density(t,p);
    
    % Initialise the saliency of the invisible vertices as a fixed value of 1.1 for a fair evaluation
    % while fine-tuning it shape by shape will lead to a significantly better performance 
    imsvsas=zeros(length(p),nViews)+1.1; 
    imsvsa=zeros(length(mesh.V),nViews);
    
    imageidxs = (i-3)*nViews+1:(i-2)*nViews;
    ims = zeros(nViews, outputSize, outputSize,3,'uint8');
    
    for j = 1:nViews
        image_name = [imagefolder '\' imagedir(imageidxs(j)+2).name];
        ims(j,:,:,:) = imread(image_name);
    end
    
    % Load the 2D saliency maps produced by the MIMO-GAN
    imsa = results_sa(imageidxs,:,:,:);
    msa= double(imsvsa);
    msa1= double(imsvsa);
    msa2= double(imsvsa);
    msa3= double(imsvsa);
    
    for ii=1:nViews
        az=viewpoints(ii,1);
        el=viewpoints(ii,2);
        
        % Load the groud truth
        if ii == 1
            gtsa=load(['.\3DVA\FixationMaps\' shapedir(i).name(1:end-4) '_300norm.txt']);
            visibility_vv=load(['.\3DVA\CentricityAndVisibilityMaps\' shapedir(i).name(1:end-4) '_300_visibility.txt']);
        elseif ii == 2
            gtsa=load(['.\3DVA\FixationMaps\' shapedir(i).name(1:end-4) '_413norm.txt']);
            visibility_vv=load(['.\3DVA\CentricityAndVisibilityMaps\' shapedir(i).name(1:end-4) '_413_visibility.txt']);
        elseif ii == 3
            gtsa=load(['.\3DVA\FixationMaps\' shapedir(i).name(1:end-4) '_599norm.txt']);
            visibility_vv=load(['.\3DVA\CentricityAndVisibilityMaps\' shapedir(i).name(1:end-4) '_599_visibility.txt']);
        end
        
        % Crop the images and prepare for the 2D-to-3D saliency transfer
        [crop,r1,r2,c1,c2]=autocrop(reshape(ims(ii,:,:,:),[224 224 3]));
        
        aa=size(crop,1);
        bb=size(crop,2);
        longside=max(aa,bb);
        shortside=min(aa,bb);
        
        image1=imsa(ii,:,:,:);
        image1_sa=reshape(image1,[224,224,3]);
        image1_sa_norm = image1_sa;
        image1_sa_norm(image1_sa<0)=0;
        image1_sa_norm = (image1_sa_norm-min(image1_sa_norm(:)))./(max(image1_sa_norm(:))-min(image1_sa_norm(:)));
        ddd = rgb2gray(image1_sa_norm);
        
        % Conduct the spatial mapping
        T=viewmtx(az,el);
        
        x4d=[p';ones(1,length(p))];
        x2d = T*x4d;
        
        xxa=max(x2d(1,:));
        xxi=min(x2d(1,:));
        yya=max(x2d(2,:));
        yyi=min(x2d(2,:));
        
        xx=xxa-xxi;
        yy=yya-yyi;
        
        longx=max(xx,yy);
        shortx=min(xx,yy);
        
        scale1=longside/longx;
        scale2=shortside/shortx;
        
        sscale=0.5*(scale1+scale2);
        x1=x2d(1,:)*sscale;
        y1=x2d(2,:)*sscale;
        
        x2=x1+0.5*outputSize;
        y2=y1-0.5*outputSize;
        
        cropsa=ddd(r1:r2,c1:c2);
        y2=-y2;
        x2=x2-min(x2);
        y2=y2-min(y2);% y2 is row;
        
        
        [vpy,vpx,vpz]=sph2cart(pi*viewpoints(ii,1)/180,pi*viewpoints(ii,2)/180,bbox);
        vpy=-vpy;
        visibility_v = mark_visible_vertices(p,t,[vpx,vpy,vpz]);
        
        [impointsx,impointsy]=meshgrid(1:bb,1:aa);
        impoints=[impointsx(:) impointsy(:)];
        
        visible=find(visibility_v~=0);
        
        vx2=x2(visible);
        vy2=y2(visible);
        x2ddd=[vx2(:) vy2(:) ];
        ind_cor = knnsearch(impoints,x2ddd);
        
        % Assign the saliency of a pixel to a visible 3D vertex
        for jj=1:length(visible)
            row=impoints(ind_cor(jj),2);
            col=impoints(ind_cor(jj),1);
            imsvsas(visible(jj),ii)=exp(1-single(W(visible(jj))))/(exp(1-cropsa(row,col)));  
        end
        
        imsvsas(:,ii)= perform_mesh_smoothing(t,p,imsvsas(:,ii));
        imsvsas(:,ii)= perform_mesh_smoothing(t,p,imsvsas(:,ii));
        
        % For efficiency, we use a simplified mesh and thus we have to map the saliency back to the original one.
        imsvsa(:,ii)=imsvsas(knng,ii);
        msa(:,ii)=imsvsa(:,ii);
        
        % As suggested by the 3DVA paper, we create 4 versions of the saliency map (0, 10, 40 and 120 smoothing iterations) 
        % and select the top-performing one for each view of each 3D object.
        
        options.niter_averaging=10; 
        imsvsa(:,ii)= perform_mesh_smoothing(mesh.F',mesh.V',imsvsa(:,ii),options);

        msa1(:,ii)=imsvsa(:,ii);
        
        options.niter_averaging=30;
        imsvsa(:,ii)= perform_mesh_smoothing(mesh.F',mesh.V',imsvsa(:,ii),options);
        msa2(:,ii)=imsvsa(:,ii);
        
        options.niter_averaging=80;
        imsvsa(:,ii)= perform_mesh_smoothing(mesh.F',mesh.V',imsvsa(:,ii),options);
        msa3(:,ii)=imsvsa(:,ii);
        
        sa=msa(:,ii);
        sa1=msa1(:,ii);
        sa2=msa2(:,ii);
        sa3=msa3(:,ii);

        sa(isnan(sa))=min(sa);
        sa1(isnan(sa1))=min(sa1);
        sa2(isnan(sa2))=min(sa2);
        sa3(isnan(sa3))=min(sa3);
        
        saa = sa1;
        saa = (saa-min(saa))/(max(saa)-min(saa));
        
        sa = sa(visibility_vv>0);
        sa = (sa-min(sa))/(max(sa)-min(sa));
        sa1 = sa1(visibility_vv>0);
        sa1 = (sa1-min(sa1))/(max(sa1)-min(sa1));
        sa2 = sa2(visibility_vv>0);
        sa2 = (sa2-min(sa2))/(max(sa2)-min(sa2));
        sa3 = sa3(visibility_vv>0);
        sa3 = (sa3-min(sa3))/(max(sa3)-min(sa3));
        
        gtsa=gtsa(visibility_vv>0);
        gtsa_norm = (gtsa-min(gtsa))/(max(gtsa)-min(gtsa));
        
        % Calculate LCC
        [RR,PP] = corrcoef(sa(:),gtsa_norm(:));
        [RR1,PP1] = corrcoef(sa1(:),gtsa_norm(:));
        [RR2,PP2] = corrcoef(sa2(:),gtsa_norm(:));
        [RR3,PP3] = corrcoef(sa3(:),gtsa_norm(:));
        RRmax = max([RR(2) RR1(2) RR2(2) RR3(2)]);
        PPmin = min([PP(2) PP1(2) PP2(2) PP3(2)]);
        cor_score(i-2,ii) = RRmax;
        p_score(i-2,ii) = PPmin;
        
        % Calculate AUC
        % As suggested by the 3DVA paper, we threshold to obtain 20% of visible vertices considered as fixations.
        numgt = round(0.2*(sum(visibility_vv>0)));
        auc = areaunderroc(gtsa_norm(:),sa(:),numgt);
        auc1 = areaunderroc(gtsa_norm(:),sa1(:),numgt);
        auc2 = areaunderroc(gtsa_norm(:),sa2(:),numgt);
        auc3 = areaunderroc(gtsa_norm(:),sa3(:),numgt);
        
        aucmax=max([auc auc1 auc2 auc3]);
        auc_score(i-2,ii) =aucmax;
        
        figure,trisurf(mesh.F',mesh.V(1,:),mesh.V(2,:),mesh.V(3,:),saa);material dull;axis equal tight;axis off;shading interp;view(az,el);
        camlight;lighting gouraud;colormap jet;
        saveas(gcf,fullfile(saliencyfolder,strcat(shapedir(i).name(1:end-4),num2str(ii),'.png')));
        
        % figure,trisurf(mesh.F',mesh.V(1,:),mesh.V(2,:),mesh.V(3,:),gtsa1_norm);material dull;axis equal tight;axis off;shading interp;
        % view(az,el);camlight;lighting gouraud;colormap jet;
        % saveas(gcf,fullfile(saliencyfolder,strcat(shapedir(i).name(1:end-4),num2str(ii),'gt.png')));
        
        close all;
    end
end

end


