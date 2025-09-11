% add manopt path
addpath(genpath('\manopt'));
clear all
close all
clc

rng(0,"twister");

% n = 5000;
n = 100;

r = ceil(sqrt(2*n));
% r = 10;
% r = 500;

% maxTime = 3; 
maxTime = 1; 

no_runs = 2;
% no_runs = 50;

opts = struct();
opts.tolgradnorm = 1e-15;
opts.maxtime = maxTime;
opts.trscache = false;
opts.maxiter = 10000;

f_bcm = cell(no_runs,1);
f_rgd = cell(no_runs,1);
f_rtr = cell(no_runs,1);
f_cg = cell(no_runs,1);
f_cg2 = cell(no_runs,1);

telapsed_bcm = cell(no_runs,1);
telapsed_rgd = cell(no_runs,1);
telapsed_rtr = cell(no_runs,1);
telapsed_cg = cell(no_runs,1);
telapsed_cg2 = cell(no_runs,1);

for k=1:no_runs
    disp("--------------current iteration------------")
    k
    
    % A = randn(n)/n;
    % A = A+A';
    % A = A + eye(n) * 50/n;
    
    A = zeros(n,n);
    for i = 1:n
        A(i,i) = (2 * randn()+500)/n;
        for j = i+1:n
            val = randn()/n;
            A(i,j) = val;
            A(j,i) = val;
        end
    end
    
    r = ceil(sqrt(2*n));

    B = randn(n,r);
    B = B ./ vecnorm(B, 2, 2);
    
    [f_bcm{k}, grad_bcm{k}, telapsed_bcm{k}, ~] = BCM(A, B, maxTime);

    [~, ~, info] = rgd(A, B', opts);
    f_rgd{k} = -[info.cost];
    telapsed_rgd{k} = [info.time];
    telapsed_rgd{k} = telapsed_rgd{k} - telapsed_rgd{k}(1);

    [~, ~, info] = rtr(A, B', opts);
    f_rtr{k} = -[info.cost];
    telapsed_rtr{k} = [info.time];
    telapsed_rtr{k} = telapsed_rtr{k} - telapsed_rtr{k}(1);

    [f_cg{k}, grad_cg{k}, telapsed_cg{k}, ~] = cg_BM(A, B, maxTime);

    eps = 0.5;
    [f_cg2{k}, grad_cg2{k}, telapsed_cg2{k}, ~] = cg_BM2(A, B, maxTime, opts, eps);

end
%%

if n >= 2000 && no_runs >= 5
    writecell(f_bcm,"f_bcm.csv")
    writecell(f_rgd, "f_rgd.csv")
    writecell(f_rtr, "f_rtr.csv")
    writecell(f_cg, "f_cg.csv")
    writecell(f_cg2, "f_cg2.csv")
    
    writecell(telapsed_bcm, "telapsed_bcm.csv")
    writecell(telapsed_rgd, "telapsed_rgd.csv")
    writecell(telapsed_rtr, "telapsed_rtr.csv")
    writecell(telapsed_cg, "telapsed_cg.csv")
    writecell(telapsed_cg2, "telapsed_cg2.csv")
end


