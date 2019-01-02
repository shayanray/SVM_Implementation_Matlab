function [ b0 , num_support_vectors] = find_b0( y_label, alpha, K, C, threshold , num_support_vectors)
    sv_index = find(alpha > threshold * max(alpha));
   
    num_support_vectors = [num_support_vectors, size(sv_index, 1)];

    b = zeros(length(sv_index),1);
    for i = 1:length(b)
        b(i) = y_label(sv_index(i)) - (alpha.*y_label)' * K(sv_index(i),:)';
    end
    b0 = mean(b);  
    
end

