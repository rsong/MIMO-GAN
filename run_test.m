
shapedir = '.\3DVA\3DModels-Simplif-up';
imagedir = '.\3DVA\3DModels-Simplif-224-up\views';
saliencydir = '.\saliency';
load('mimogan.mat');
[cor_score,p_score,auc_score] = sa_trans_eval(shapedir, imagedir, saliencydir, results_sa);

