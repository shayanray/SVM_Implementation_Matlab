function [y_pred , y_copy, W ] = predict(W, X, y)
    
    
[~,y_pred] =  max(W' * X); % probability and max column, need max column only
y_pred = y_pred';
y_copy = y;

end