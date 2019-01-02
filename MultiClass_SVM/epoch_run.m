function [total_loss_hstry , sumW, W ] = epoch_run(X, y, eta_0, eta_1, C, W, total_epochs,total_loss_hstry)
    

    for i = 1 : total_epochs
        
        fprintf('Running Epoch# %d: \n', i);
        learning_rate = eta_0/(eta_1 + i);
        count_X = size(X,2); % number of samples
        [loss_trn, sumW, W] = find_loss(X, y, learning_rate, C, count_X, W);
        total_loss_hstry = [total_loss_hstry loss_trn]; %reshape(hist_loss_trn.',1,[])
    end

end
