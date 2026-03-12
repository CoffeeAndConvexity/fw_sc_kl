function [f_val, grad_val, telapsed, B] = cg_BM2(A,B,maxTime,opts,eps)
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
    if grad_val(ep) > eps
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
    else
        flag = false;
        break
    end
end

f_val = f_val(1:ep);
grad_val = grad_val(1:ep);
telapsed = telapsed(1:ep);

if ~flag
    opts.maxtime = maxTime - telapsed(ep);
    [B, xcost, info] = rtr(A, B', opts);
    B = B';
    f_val_rtr = -[info.cost];
    f_val_rtr = f_val_rtr(2:end)';
    grad_val_rtr = [info.gradnorm];
    grad_val_rtr = grad_val_rtr(2:end)';
    telapsed_rtr = [info.time];
    telapsed_rtr = telapsed_rtr - telapsed_rtr(1);
    telapsed_rtr = telapsed(ep) + telapsed_rtr(2:end)';
    
    f_val = [f_val; f_val_rtr]';
    grad_val = [grad_val; grad_val_rtr]';
    telapsed = [telapsed; telapsed_rtr]';
end
end