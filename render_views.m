function [num_vertices,ims,viewpoints] = render_views( mesh, varargin )
%RENDER_VIEWS render a 3d shape from multiple views
%   mesh::
%       a mesh object containing fileds
%           .F 3 x #faces (1-based indexing)
%           .V 3 x #vertices
%       OR a path to .off file
%   `az`:: (default) [0:30:330]
%       horizontal viewing angles, use this setting for shapes that are
%       upright oriented according to +Z axis!
%   `el`:: (default) +/-30
%       vertical elevation, , use this setting for shapes that are
%
%   `colorMode`:: (default)  'rgb'
%       color mode of output images ('rgb' or 'gray')
%   `outputSize`::  (default)  224
%       output image size (both dimensions)
%   `minMargin`:: (default)  0.1
%       minimun margin ratio in output images
%   `maxArea`:: (default)  0.3
%       maximun area ratio in output images
%   `figHandle`:: (default) []
%       handle to existing figure

opts.views = 0:30:330;
opts.views=repmat(opts.views,[1 2]);
opts.views=repmat(opts.views,[2 1]);
opts.views=opts.views';
opts.views(1:12,2)=30;
opts.views(13:24,2)=-30;

opts.colorMode = 'rgb';
opts.outputSize = 224;
opts.minMargin = 0.1;
opts.maxArea = 0.3;
opts.figHandle = [];
opts = vl_argparse(opts,varargin);

if isempty(opts.figHandle)
    opts.figHandle = figure;
end

if ischar(mesh)
    if strcmpi(mesh(end-2:end),'off') || strcmpi(mesh(end-2:end),'obj')
        mesh = loadMesh(mesh);
    else
        error('file type (.%s) not supported.',mesh(end-2:end));
    end
end

num_vertices=length(opts.views);
viewpoints=opts.views;

ims = cell(1,length(opts.views));
for i=1:length(opts.views)
    
    plotMesh(mesh,'solid',opts.views(i,1),opts.views(i,2));
    ims{i} = print('-RGBImage', '-r100'); %In case of an error or an old matlab version: comment this line and uncomment the following 2 ones
    %saveas(opts.figHandle, '__temp__.png');
    %ims{i} = imread('__temp__.png');
    if strcmpi(opts.colorMode,'gray'), ims{i} = rgb2gray(ims{i}); end
    ims{i} = resize_im(ims{i}, opts.outputSize, opts.minMargin, opts.maxArea);
end
end


function im = resize_im(im,outputSize,minMargin,maxArea)

max_len = outputSize * (1-minMargin);
max_area = outputSize^2 * maxArea;

nCh = size(im,3);
mask = ~im2bw(im,1-1e-10);
mask = imfill(mask,'holes');
% blank image (all white) is outputed if not object is observed
if isempty(find(mask, 1))
    im = uint8(255*ones(outputSize,outputSize,nCh));
    return;
end
[ys,xs] = ind2sub(size(mask),find(mask));
y_min = min(ys); y_max = max(ys); h = y_max - y_min + 1;
x_min = min(xs); x_max = max(xs); w = x_max - x_min + 1;
scale = min(max_len/max(h,w), sqrt(max_area/sum(mask(:))));
patch = imresize(im(y_min:y_max,x_min:x_max,:),scale);
[h,w,~] = size(patch);
im = uint8(255*ones(outputSize,outputSize,nCh));
loc_start = floor((outputSize-[h w])/2);
im(loc_start(1)+(0:h-1),loc_start(2)+(0:w-1),:) = patch;

end
