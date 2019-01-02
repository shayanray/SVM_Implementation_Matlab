function [y_pred , y_copy, W ] = predict(W, X, y)
    
    
[~,y_pred] =  max(W' * X); % probability and max column, need max column only
y_pred = y_pred';
y_copy = y;

%use this to match accuracy since -1 is not a valid index. considering -1
%equivalent to 10
for i = 1 : size(X,2)
    if y(i) == -1
        y_copy(i) = 2;
    end
end

end