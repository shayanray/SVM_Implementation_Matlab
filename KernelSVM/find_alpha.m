function [ alpha ] = find_alpha( num_samples, y_lbl, K, C )
    %% Initialize inputs for quadprog

    H = zeros(num_samples, num_samples);
    for i = 1:num_samples
        for j = 1:num_samples
            H(i,j) = y_lbl(i) * y_lbl(j) * K(i,j);
        end
    end
    f = -ones(num_samples, 1);
    Aeq = y_lbl';
    Beq = 0;
    lb = zeros(num_samples, 1);
    ub = ones(num_samples, 1) * C;
    x0 = [];
    options = optimset('Algorithm','interior-point-convex','Display','off','MaxIter', 10000);
    A = [];
    b = [];
    
    %% Call quadprog
    [ alpha, fVal, exitFlag ] = quadprog(H, f, A, b, Aeq, Beq,lb, ub, x0, options);

end


