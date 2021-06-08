function render_designated_views_in_a_folder(folder, varargin)
% calls 'render_designated_views' for every shape found in the given folder

opts.ext = '.jpg';          % extension of target files
opts.range = [];            % if empty, all found shapes will be rendered, while a range [X:Y] will render shapes in the given range

opts = vl_argparse(opts, varargin);

mesh_filenames = [rdir( sprintf('%s\\*.obj', folder) ); rdir( sprintf('%s\\*.off', folder) )];

if isempty( opts.range )
    range = 1:length( mesh_filenames );
else
    range = opts.range;
end

fig = figure('Visible','off');
for fi=range
    modelname = mesh_filenames(fi).name;
    
    % Replace the following line with the appropriate one.
    viewpoints = load(['D:\MATLAB\R2019a\work\3dsaliency\test_viewpoints\' modelname(54:end-4) '.txt']);
    
    fprintf('Loading and rendering input shape %s...', mesh_filenames(fi).name );
    mesh = loadMesh( mesh_filenames(fi).name );
    if isempty(mesh.F)
        error('Could not load mesh from file');
    else
        fprintf('Done.\n');
    end
    
   ims = render_designated_views(mesh, 'figHandle', fig, 'views', viewpoints);
    for ij=1:length(ims)
        imwrite( ims{ij}, sprintf('%s_%03d%s', mesh_filenames(fi).name(1:end-4), ij, opts.ext), 'Quality',100 );
    end
end
close(fig);
