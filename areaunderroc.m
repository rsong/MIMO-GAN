function auc=areaunderroc(gt,sa,numgt)
gt=(gt-min(gt))/(max(gt)-min(gt));
sa=(sa-min(sa))/(max(sa)-min(sa));

% gt_thres=0.02;
sa_thres=0:0.01:1;
gt_des=sort(gt,'descend');
gt_bin=zeros(length(gt),1);
gt_sep=gt_des(numgt);
gt_bin(gt>=gt_sep)=1;

tprs=zeros(length(sa_thres),1);
fprs=zeros(length(sa_thres),1);

for i=1:length(sa_thres)
    curr_thres=sa_thres(i);
    
    sa_bin=zeros(length(sa),1);
    sa_bin(sa>curr_thres)=1;
    tpr=sum(gt_bin(sa_bin==1))/sum(gt_bin);
    tprs(i)=tpr;
    
    tnr=sum(abs(gt_bin(sa_bin==0)-1))/sum(gt_bin==0);
    fpr=1-tnr;
    fprs(i)=fpr;   
end
fprs= fprs(end:-1:1); %fliplr
tprs = tprs(end:-1:1); %fliplr

auc=trapz(fprs,tprs);
if isnan(auc)
    auc=0;
end
end