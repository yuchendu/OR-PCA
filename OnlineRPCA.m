function [ORPCA_model, residual] = OnlineRPCA(X_ab,X_nor, param)

lambda1 = param.lambda1;
lambda2 = param.lambda2;

num_trained_normal = param.num_trained_normal;

converge_error = param.error;
algorithm_iterMax = param.Algorithm_iterMax;

sz_X_ab = size(X_ab);
n = size(X_ab,3);
p = size(X_ab,1) * size(X_ab,2);
r = param.r;

% initialize X
% X: tensor -> matrix, each column represents an image
X_ab = double(tenmat(X_ab,3)');

% normalization, optional
% mean_square_X = sum(X .* X, 1);
% X = X ./ repmat(mean_square_X, size(X,1), 1);

% initialize the dictionary (low-rank basis)
if ~isempty(X_nor)
    % use normal fundus images as U
    X_nor = double(tenmat(X_nor,3)');
    % randomly select r atoms from normal image set
    U = X_nor(:,randperm(size(X_nor,2), r));
else
    % if 
    U = rand(p,r);
end

% initialize the sparse error S, reconstructed low-rank image L
S = zeros(p,n);
L = zeros(p,n);
residual = zeros(num_trained_normal+n,1);

% initialize the interval variables A and B
A = zeros(r,r);
B = zeros(p,r);

X = cat(2,X_nor(:,randperm(size(X_nor,2), num_trained_normal)),X_ab);
% train the model
for i = 1:(num_trained_normal+n)
    if i <= num_trained_normal
        disp(['## The ', num2str(i),...
            'th online decomposition of normal image started, totally ',num2str(num_trained_normal+n),...
            ' images. ##']);
    else
        disp(['## The ', num2str(i-num_trained_normal),...
            'th online decomposition of abnormal image started, totally ',num2str(num_trained_normal+n),...
            ' images. ##']);
    end
    
    x_i = X(:,i);
    s_i = zeros(size(X,1),1);
    U_i = U;
    
    Algorithm_converge = 0;
    iteration_time = 0;
    while ~Algorithm_converge
        iteration_time = iteration_time + 1;
        v_i = (U_i'*U_i + lambda1*eye(size(U_i,2)))\U_i'*(x_i - s_i);
        s_i = softThreshold(x_i - U_i*v_i, lambda2);
        l_i = U_i*v_i;
        error_converge = norm(x_i-l_i-s_i,2)/norm(x_i);
        if error_converge <= converge_error
            disp('# Algorithm converged. #');
            disp(['U max: ', num2str(max(U_i(:))),', x_i - U*v_i, max: ',num2str(max(x_i - U_i*v_i))]);
            Algorithm_converge = 1;
        end
        if iteration_time >= algorithm_iterMax
            disp('# Maximum iteration reached. #');
            disp(['U max: ', num2str(max(U_i(:))),', x_i - U*v_i, max: ',num2str(max(x_i - U_i*v_i))]);
            break;
        end
    end
    residual(i) = error_converge;
    s_i_diff = x_i - l_i;

    im_x = uint8(reshape(x_i,[sz_X_ab(1), sz_X_ab(2)]));
    im_l = uint8(reshape(l_i,[sz_X_ab(1), sz_X_ab(2)]));
    im_s = uint8(abs(reshape(s_i,[sz_X_ab(1), sz_X_ab(2)])));
    im_s_diff = uint8(abs(reshape(s_i_diff,[sz_X_ab(1), sz_X_ab(2)])));

    figure(1);
    subplot(2,2,1);
    imshow(im_x);
    title(['The original image']);
    subplot(2,2,2);
    imshow(im_l);
    title(['The low-rank image']);
    subplot(2,2,3);
    imshow(im_s);
    title(['The sparse image directly obtained']);
    subplot(2,2,4);
    imshow(im_s_diff);
    title(['The sparse image from x-l']);
    if i <= num_trained_normal
        suptitle(['The ', num2str(i),'th normal image in X']);
    else
        suptitle(['The ', num2str(i-num_trained_normal),'th abnormal image in X']);
    end
    drawnow;


    A = A + v_i * v_i';
    B = B + (x_i - s_i) * v_i';
    
    disp(['A: ',num2str(min(A(:))),'~',num2str(max(A(:))), ', B: ',num2str(min(B(:))),'~',num2str(max(B(:)))]);

    A_wave = A+lambda1*eye(size(A,2));

    U = U_i;
    for jj = 1:size(U,2)
        U(:,jj) = U(:,jj) - (U * A_wave(:,jj) - B(:,jj))/A_wave(jj,jj);
    end
    
    if i <= num_trained_normal
        disp(['## The ', num2str(i),...
            'th normal image has been finished, reconstruct error: ',num2str(error_converge), '. ##']);
    else
        disp(['## The ', num2str(i-num_trained_normal),...
            'th abnormal image has been finished, reconstruct error: ',num2str(error_converge), '. ##']);
    end
    disp('---------------');
     
    if i <= num_trained_normal 
        continue; 
    end
    
    L(:,i-num_trained_normal) = l_i;
    S(:,i-num_trained_normal) = s_i;
end

ORPCA_model.L = matten(L', sz_X_ab, 3);
ORPCA_model.S = matten(S', sz_X_ab, 3);
end

function [L] = softThreshold(Mat,mu)
% S_epsilon[X] = argmin mu*||X||_1 + 1/2*||X-Y||_fro2
% S_epsilon[X] = sgn(W) * max(|W|-mu,0)
MatThresh = abs(Mat)-mu;
MatThresh(MatThresh<0) = 0;
L = sign(Mat) .* MatThresh;
% rank
% diagS = diag(L);
% svp = length(find(diagS > 1/mu));
end
