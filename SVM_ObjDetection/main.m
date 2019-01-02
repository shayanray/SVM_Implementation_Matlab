%% Run HW2_Utils.getPosAndRandomNeg(); in matlab command line to generate 
%../hw2data/trainval_random.mat'
% inside stochastic svm(a copy of Q3.2) load ../hw2data/trainval_random.mat'
%% get W from stochastic_svm (copy of solution 3.2)

W_valdn = stochastic_svm();
b = 0;
results = './4.4.1-output.mat';
HW2_Utils.genRsltFile(W_valdn, b, 'val', results);

[ap, prec, rec] = HW2_Utils.cmpAP('4.4.1-output.mat', 'val');

 %% print outputs

    fprintf('******************** R E S U L T S *************************\n');
    fprintf('AP = %d \n', ap);
    %fprintf('Precision >> %d \n', prec);
    %fprintf('Recall >> %d \n', rec);
    