%% Answer 3.1 - kernel SVM using Quadratic Programming

%checw for data file
if exist('input/q3_1_data.mat', 'file') == 0
    error(['Need q3_1_data.mat file in the INPUT directory']);
end
disp('Loading q3_data.mat now .. ');


%% Load file q3_1_data.mat
load 'input/q3_1_data.mat';
X_train = trD;
y_train = trLb;
X_valdtn = valD;
y_valdtn = valLb;

%% Variables to configure

p = 0;                  %linear SVM (0)
C = 10;                 %C, vary between 1/10 and 1.0
threshold = 0.0005;     %define sv
num_supprt_vec =[];

%% Compute the kernel(w)
w_trn = find_w(X_train, X_train, p);

%% Calculate alpha
num_samples_trn = length(X_train(1,:));
alpha_trn = find_alpha(num_samples_trn, y_train, w_trn, C);

%% Calculate b
[b, num_supprt_vec] = find_b0(y_train, alpha_trn, w_trn, C, threshold, num_supprt_vec);

%% Calculate predicted value for training data
y_train_pred = sum(bsxfun(@times, w_trn, (alpha_trn .* y_train)') , 2) + b * ones(num_samples_trn, 1);
accu_trn = mean(sign(y_train_pred) == y_train);
trn_objective_val = sum(sign(y_train_pred) == y_train);

%% Compute the w for valdtn set
w_vldn = find_w(X_valdtn, X_valdtn, p);

num_samples_tst = length(X_valdtn(1,:));
alpha_tst = find_alpha(num_samples_tst, y_valdtn, w_vldn, C);
%% Calculate predicted value for valdtn data
y_vald_pred = sum(bsxfun(@times, w_vldn, (alpha_tst .* y_valdtn)') , 2) + b * ones(num_samples_tst, 1);

%% valdtn accuracy, confusion matrix

accu_tst = mean(sign(y_vald_pred) == y_valdtn);
valdn_objective_val = sum(sign(y_vald_pred) == y_valdtn);

confuMat = confusionmat(y_valdtn,y_vald_pred);

%% Publish Confusion matrix in CSV

if C == 0.1
    csvwrite('output/confusionMatrix0.1.csv',confuMat);
end

if C == 10.0
    csvwrite('output/confusionMatrix10.csv',confuMat);
end
%% Report the output parameters

fprintf('For C = >> %d \n', C);
fprintf('Train Accuracy >> %d \n', accu_trn);
fprintf('valdtn Accuracy >> %d \n', accu_tst);
fprintf('Train Objective Val  >> %d \n', trn_objective_val);
fprintf('Validaton Objective Val  >> %d \n', valdn_objective_val);
fprintf('Number of support vectors >> %d \n', num_supprt_vec);
fprintf('Confusion Matrix CSV is generated in output folder.  \n');
fprintf('For C = 0.1 >> confusionMatrix0.1.csv  \n');
fprintf('For C = 10.0 >> confusionMatrix10.csv  \n ');
