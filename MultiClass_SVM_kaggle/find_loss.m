function [one_epoch_loss , sumW, W ] = find_loss(features, lbl, learningRate, C, count_X, W)


    %% Initialize inputs
    
    num_train = size(features, 1); 
    num_classes = length(unique(lbl));
    
    prev_loss = 1000;
    threshold = 0.0000000111;
    
    one_epoch_loss = 0;
    for i = randperm(count_X) %permute the index numbers
        
        x_i = features(:, i);
        
        %considering -1 equivalent to 10 as -1 causes issues
        y_i = lbl(i);
        
        tmpW = W;
        tmpW(:, y_i) = -inf; 
        
        [~,y_hat]=max(tmpW'*x_i);
        loss_param = W(:, y_hat)'*x_i - W(:, y_i)'*x_i + 1;
        
        for j = 1:num_classes
            if loss_param > 0
                if( j == y_i)  %y_i
                    W(:,j) = W(:,j) - learningRate*((W(:, y_i))./count_X  - C.*x_i);  
                elseif(j == y_hat) %y_i_hat
                    W(:,j) = W(:,j) - learningRate* ((W(:, y_hat))./count_X  + C.*x_i);
                else
                    W(:,j) = W(:,j) - learningRate* (W(:, j))./count_X;
                end    
            else
                W(:,j) = W(:,j) - learningRate*(W(:, j))./count_X ;
            end
            
          % check if loss has converged then break  
            if (mod(i, 50) == 0)
                %fprintf('Instance %d / %d: loss %f \n', i, count_X, loss_param);
                if (abs(prev_loss - loss_param) > threshold )
                    prev_loss = loss_param;
                else
                    break;
                end
       
            end
      
        end
        
        hinge_loss = max((W(:,y_hat)'*x_i-W(:,y_i)'*x_i+1),0); % max loss
        one_epoch_loss = one_epoch_loss + (sum(vecnorm(W).^2))/(2*count_X) + C*hinge_loss;
    end
    
    sumW = sum(vecnorm(W).^2); 
end


