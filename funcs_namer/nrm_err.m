function [ nerr ] = nrm_err(alpha, x1, x2 )

alpha = alpha(1) + 1i * alpha(2);

nerr = norm(x1(:) * alpha - x2(:)) / norm(x2(:));
