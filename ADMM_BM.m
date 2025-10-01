function [f_val, grad_val, telapsed, B] = ADMM_BM(A,B,maxTime)
maxIter = 10000;
[n,r] = size(B);
f_val = zeros(maxIter+1,1);
grad_val = zeros(maxIter+1,1);
telapsed = zeros(maxIter+1,1);

rho = norm(A,"fro");
rho_inv = 1/rho;
ep = 1;
g = A*B;
f_val(ep) = sum(sum(A*B.*B));
grad_val(1) = 2*sqrt(sum(sum(g.*g))-sum(sum(B.*g,2).^2));
lamda = ones(n,1);
Beta = B;
Y = A * B;
while true
    tstart = tic;
    B = Beta - rho_inv*(Y+ A * Beta);
    lamda = vecnorm(B, 2, 2);
    B = B ./ lamda;
    Beta = B + rho_inv * (Y - A * B);
    Y = Y + rho * (B - Beta);
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