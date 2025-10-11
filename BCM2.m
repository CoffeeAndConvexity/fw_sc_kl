function [ f_val, grad_val, telapsed, B ] = BCM2(A, B, maxTime, opts, eps)
[n, ~] = size(A);
maxIter = 10000;
f_val = zeros(maxIter+1,1);
grad_val = zeros(maxIter+1,1);
telapsed = zeros(maxIter+1,1);

ep = 1;
g = A*B;
f_val(1) = sum(sum(g.*B));
grad_val(1) = 2*sqrt(sum(sum(g.*g))-sum(sum(B.*g,2).^2));

while true
    if grad_val(ep) > eps
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
            flag = true;
            break
        end
    else
        flag = false;
        break
    end
end

f_val = f_val(1:ep)';
grad_val = grad_val(1:ep)';
telapsed = telapsed(1:ep)';

if ~flag
    opts.maxtime = maxTime - telapsed(ep);
    [sigma, xcost, info] = rtr(A, B', opts);
    sigma = sigma';
    f_val_rtr = -[info.cost];
    f_val_rtr = f_val_rtr(2:end)';
    grad_val_rtr = [info.gradnorm];
    grad_val_rtr = grad_val_rtr(2:end)';
    telapsed_rtr = [info.time];
    telapsed_rtr = telapsed_rtr - telapsed_rtr(1);
    telapsed_rtr = telapsed(ep) + telapsed_rtr(2:end)';
    
    f_val = [f_val'; f_val_rtr]';
    grad_val = [grad_val'; grad_val_rtr]';
    telapsed = [telapsed'; telapsed_rtr]';
end

end

