%% Answer 3.2 - Stochastic Gradient Descent for Multi-Class SVM

%check for data file
if exist('input/q3_2_data.mat', 'file') == 0
    error(['Need q3_2_data.mat file in the INPUT directory']);
end
disp('Loading q3_data.mat now .. ');

%% Load file q3_2_data.mat
load 'input/q3_2_data.mat';

X_train = trD;
y_train = trLb;
X_valdn = valD;
y_valdn = valLb;
X_test = tstD;

%% All Variables to be configured for train and validation

C = 0.05;    %margin 0.10, 10
eta_0 = 1;
eta_1 = 10;
total_epochs = 2000;
total_hist_trn_loss = [];
total_hist_valdn_loss = [];

%% training data epoch run
disp('Starting Execution on Training data..............');
class_cnt = max(y_train(:));
num_train = size(X_train, 1); %features
W_train = zeros(num_train,class_cnt); % initial W set to 0
[total_hist_trn_loss , sumW_train, W_train ] = epoch_run(X_train, y_train, eta_0, eta_1, C, W_train, total_epochs, total_hist_trn_loss);


%% training data prediction and accuracy
[y_pred_trn , y_copy_trn, W ] = predict(W_train, X_train, y_train);
train_accuracy = mean((y_pred_trn) == y_copy_trn);
train_obj_val = sum((y_pred_trn) == y_copy_trn);

%% Visualize Training plot
figure, plot(total_hist_trn_loss);
xlabel('Num Epochs');
ylabel('Training Loss');


%% validation data epoch run
disp('Starting Execution on Validation data..............');
class_cnt = max(y_valdn(:));
num_valdn = size(X_valdn, 1); %features
W_valdn = zeros(num_valdn,class_cnt); % initial W set to 0
[total_hist_valdn_loss , sumW_valdn, W_valdn ] = epoch_run(X_valdn, y_valdn, eta_0, eta_1, C, W_valdn, total_epochs, total_hist_valdn_loss);


%% validation data prediction and accuracy
[y_pred_valdn , y_copy_valdn, W ] = predict(W_valdn, X_valdn, y_valdn);

valdn_accuracy = mean((y_pred_valdn) == y_copy_valdn);
valdn_obj_val = sum((y_pred_valdn) == y_copy_valdn);

%% Compute plot
figure, plot(total_hist_valdn_loss);
xlabel('Num Epochs');
ylabel('Validation Loss');

%% Predict Test results using W from Validation
[~,y_pred_test] =  max(W_valdn' * X_test); % probability and max column
y_pred_test = y_pred_test';


index = [];
for i = 1:length(y_pred_test)
    index = [index; i, round(y_pred_test(i))];
end

% write to CSV
T = array2table(index,'VariableNames',{'Id','Class'});
writetable(T,'output/kaggle-Submission.csv');


%% Publish Confusion matrix in CSV

if C == 0.1
    csvwrite('output/train_error_0.1.csv',total_hist_trn_loss');
    csvwrite('output/test_error_0.1.csv',total_hist_valdn_loss');
end

if C == 10.0
    csvwrite('output/train_error_10.csv',total_hist_trn_loss');
    csvwrite('output/test_error_10.csv',total_hist_valdn_loss');
end

%% print outputs

fprintf('******************** R E S U L T S *************************\n');
fprintf('For C = >> %d \n', C);
fprintf('Train Accuracy >> %d \n', train_accuracy);
fprintf('Train Objective Val  >> %d \n', train_obj_val);

fprintf('Validation Accuracy >> %d \n', valdn_accuracy);
fprintf('Validation Objective Val  >> %d \n', valdn_obj_val);

fprintf('W train sum >> %d \n', sumW_train);
fprintf('W Validation sum  >> %d \n', sumW_valdn);

fprintf('Confusion Matrix CSV is generated in ----OUTPUT----- folder.  \n');

fprintf('************************************************************\n');
