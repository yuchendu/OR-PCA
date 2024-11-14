% clear;
% clc;
if exist('store','var')
    last = store;
end

path_ab_K = '..\TensorDecomposition\EvaluationMethod\193\193\(436-422)RemVesVesRGB(193)changename';
path_ab_M = '..\TensorDecomposition\EvaluationMethod\110\110\Kaggle+MessidorLesionAlignCutRemVesNormColorG20171026(changename)';
[ImgTensor_ab_K,file_ab_K] = fileloading(path_ab_K,2,'*.bmp');
ImgTensor_ab_K = double(ImgTensor_ab_K);

ImgTensor_ab = ImgTensor_ab_K;

path_nor = '..\TensorDecomposition\imageNormal';
[ImgTensor_nor,~] = fileloading(path_nor,2,'*.bmp');
ImgTensor_nor = double(ImgTensor_nor);


%%
% parameters setting for MoGRPCA_inexact
result = cell(10,10);
for ii = 1:1
for jj = 1:1

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
path_groundtruth_K = '..\TensorDecomposition\groundtruth\193_org_num';
path_groundtruth_M = '..\TensorDecomposition\groundtruth\110_org_num';
[groundtruthTensor_K,~] = fileloading(path_groundtruth_K,1,'*.bmp');
[groundtruthTensor_M,~] = fileloading(path_groundtruth_M,1,'*.bmp');
% groundtruthTensor = cat(3,groundtruthTensor_K,groundtruthTensor_M);
groundtruthTensor = groundtruthTensor_K;
[auc,pr,FPR,SE,PPv] = AUC_PR(groundtruthTensor,Output_F);
disp(['lambda1__',num2str(param.lambda1),...
    '__lambda2_',num2str(param.lambda2),...
    '__dictSize_',num2str(param.num_atoms),...
    '__AUC__',num2str(auc),'__AP_',num2str(pr)]);
store = ['lambda1__',num2str(param.lambda1),...
    '__lambda2_',num2str(param.lambda2),...
    '__dictSize_',num2str(param.num_atoms),...
    '__AUC__',num2str(auc),'__AP_',num2str(pr)];
result{ii,jj} = store;

%%
% save([store,'.mat'],'Output_A','Output_E','mog_model','FPR','SE','PPv');
save([store,'.mat'],'FPR','SE','PPv');
end
end