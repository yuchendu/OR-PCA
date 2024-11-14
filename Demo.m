% clear;
% clc;
if exist('store','var')
    last = store;
end

path_ab = 'your_folder_path';
[ImgTensor_ab,file_ab] = fileloading(path_ab,2,'*.bmp');
ImgTensor_ab = double(ImgTensor_ab);

path_nor = 'your_folder_path';
[ImgTensor_nor,~] = fileloading(path_nor,2,'*.bmp');
ImgTensor_nor = double(ImgTensor_nor);


%%
% parameters setting for MoGRPCA_inexact
param.lambda1 = 15;
param.lambda2 = 3;
param.r = 250;
param.num_trained_normal = 300;
param.Algorithm_iterMax = 10;
param.error = 1e-2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ORPCA_model, residual] = OnlineRPCA(ImgTensor_ab, ImgTensor_nor, param);
Output_B = ORPCA_model.L;
Output_F = ORPCA_model.S;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
figure();
plot(1:size(residual,1),residual);
path_groundtruth = 'your_folder_path';
[groundtruthTensor,~] = fileloading(path_groundtruth,1,'*.bmp');
[auc,pr,FPR,SE,PPv] = AUC_MAP(groundtruthTensor,Output_F);
disp(['lambda1__',num2str(param.lambda1),...
    '__lambda2_',num2str(param.lambda2),...
    '__dictSize_',num2str(param.r),...
    '__trainedNormalImages_',num2str(param.num_trained_normal),...
    '__AUC__',num2str(auc),'__AP_',num2str(pr)]);
store = ['lambda1__',num2str(param.lambda1),...
    '__lambda2_',num2str(param.lambda2),...
    '__dictSize_',num2str(param.r),...
    '__trainedNormalImages_',num2str(param.num_trained_normal),...
    '__AUC__',num2str(auc),'__AP_',num2str(pr)];

%%
% save([store,'.mat'],'Output_B','Output_F','mog_model','FPR','SE','PPv');
% save([store,'.mat'],'FPR','SE','PPv');
