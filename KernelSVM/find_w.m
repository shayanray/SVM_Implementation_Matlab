function [ w ] = find_w(input_data1, input_data2, p)
% Calculates kernel(w)

    M = length(input_data1(1,:));
    N = length(input_data2(1,:));
    w = zeros(M, N);
    for i = 1:M
        for j = 1:N
            w(i,j) = input_data1(:,i)' * input_data2(:,j);
        end
    end

end

