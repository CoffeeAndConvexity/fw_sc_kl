function [f_val, grad_val, telapsed, B] = cg_BM(A,B,maxTime)
maxIter = 10000;
[n,r] = size(B);
f_val = zeros(maxIter+1,1);
grad_val = zeros(maxIter+1,1);
telapsed = zeros(maxIter+1,1);

ep = 1;
g = A*B;
f_val(ep) = sum(sum(A*B.*B));
grad_val(1) = 2*sqrt(sum(sum(g.*g))-sum(sum(B.*g,2).^2));
lamda = ones(n,1);
while true
    tstart = tic;
    B = A * B;
    lamda = vecnorm(B, 2, 2);
    B = B ./ lamda;
    tepoch = toc(tstart);
    ep = ep+1;
    g = A*B;
    f_val(ep) = sum(sum(g.*B));
    grad_val(ep) = 2*sqrt(sum(sum(g.*g))-sum(sum(B.*g,2).^2));
    telapsed(ep) = telapsed(ep-1) + tepoch;

    if telapsed(ep) > maxTime
        flag = true;
        break
    end
end

f_val = f_val(1:ep)';
grad_val = grad_val(1:ep)';
telapsed = telapsed(1:ep)';
end