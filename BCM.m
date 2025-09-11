function [ f_val, grad_val, telapsed, B ] = BCM(A, B, maxTime)
maxIter = 10000;
[n,r] = size(B);
f_val = zeros(maxIter+1,1);
grad_val = zeros(maxIter+1,1);
telapsed = zeros(maxIter+1,1);

ep = 1;
g = A*B;
f_val(ep) = sum(sum(A*B.*B));
grad_val(1) = 2*sqrt(sum(sum(g.*g))-sum(sum(B.*g,2).^2));

while true
    tstart = tic;
    for i=1:n
        g_i = A(i,:)* B;
        B(i,:) = g_i/norm(g_i);
    end
    tepoch = toc(tstart);
    
    ep = ep+1;
    g = A*B;
    f_val(ep) = sum(sum(g.*B));
    grad_val(ep) = 2*sqrt(sum(sum(g.*g))-sum(sum(B.*g,2).^2));
    telapsed(ep) = telapsed(ep-1) + tepoch;

    if telapsed(ep) > maxTime
        break
    end
end

f_val = f_val(1:ep)';
grad_val = grad_val(1:ep)';
telapsed = telapsed(1:ep)';

end

