# MIMO-GAN

The Python codes were developed based on Python 3.7.7 and Tensorflow 2.1.0.
The MATLAB codes were developed based on MATLAB 2019a.

To test the pretrained MIMO-GAN on the 3DVA dataset, implement the following steps:
1. Download the 3DVA dataset from https://drive.google.com/file/d/1LQByw2m1l_kywi95_tFhVg-XUuA7jrP7/view?usp=sharing or the webpage provided in original EG18 paper and unzip it. Note that here we also provided the rendered 2D views at the designated views for all meshes of the 3DVA dataset. Alternatively, you can render them by using the MATLAB function 'render_designated_views_in_a_folder.m'. In the case that you want to do it for a single mesh, you can call 'render_designated_views'.
2. Download the pretrained MIMO-GAN model (~133MB H5 file) from 
https://drive.google.com/file/d/1tkgxOfDFAKUzQq8OLaqOhOcnWrBwb4iL/view?usp=sharing
3. Implement 'h5tomat.py' to generate the predicted 2D saliency for all meshes of the 3DVA dataset, which outputs 'mimogan.mat' required for running 'run_test.m' in MATLAB. Alternatively, you can directly download 'mimogan.mat' from https://drive.google.com/file/d/1E8zbGJMjsfPGyNKCIckimsCtFbiZXL8h/view?usp=sharing
4. Create a folder called 'saliency'.
5. Run 'run_test.m' in MATLAB. It will output the view-dependent mesh saliency maps in the 'saliency' folder and output the LCC and AUC scores.

To train the MIMO-GAN using SALICON and ModelNet40 datasets, implement the following steps:
1. Download the SALICON dataset from the official webpage.
2. Download the ModelNet40 dataset (the up-oriented version) from the official webpage.
3. Call 'render_views_of_all_meshes_in_a_folder.m' in MATLAB to generate the projected 2D views.
4. Run 'mimogan.py' to train MIMO-GAN. Please make sure that the paths of the training datasets are set correctly in lines 18-20:
train_data_dir contains the 2D natural images from SALICON
label_data_dir contains the corresponding fixation maps from SALICON
object_data_dir contains the projected 2D views of the 3D meshes from ModelNet40, which thus includes 40 subfolders.


If you use our codes, please cite:
Ran Song, Wei Zhang, Yitian Zhao, Yonghuai Liu and Paul L. Rosin. Mesh Saliency: An Independent Perceptual Measure or A Derivative of Image Saliency? CVPR 2021
