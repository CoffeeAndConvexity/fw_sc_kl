addpath(genpath('C:\Users\kramn\Desktop\USELESS\Okayeg\research\manopt'));
clear all
close all
clc

% add CGAL codes, note that we had to change the code a little bit to
% compute feasible objective and enforce REAL WALLTIME, the code itself
% uses totCpuTime but control message says "walltime achieved" not to
% mention it also burdens itself with in between logging computations that
% are not solver related in its original form. 
addpath utils;
addpath solver;

s1 = RandStream('twister','Seed',0);

n = 20000;

maxNumCompThreads(16)
r = ceil(sqrt(2*n));

maxTime = 60; 
no_runs = 50;
% % 
% maxTime = 2; 
% no_runs = 2;

opts = struct();
opts.tolgradnorm = 1e-15;
opts.maxtime = maxTime;
opts.trscache = false;
opts.maxiter = 10000;

f_bcm2 = cell(no_runs,1);

f_bcm = cell(no_runs,1);
f_rgd = cell(no_runs,1);
f_rtr = cell(no_runs,1);
f_cg = cell(no_runs,1);
f_cg2 = cell(no_runs,1);

f_cgal1 = cell(no_runs,1);
f_cgal2 = cell(no_runs,1);
f_admm  = cell(no_runs,1);


telapsed_bcm2 = cell(no_runs,1);

telapsed_bcm = cell(no_runs,1);
telapsed_rgd = cell(no_runs,1);
telapsed_rtr = cell(no_runs,1);
telapsed_cg = cell(no_runs,1);
telapsed_cg2 = cell(no_runs,1);

telapsed_cgal1 = cell(no_runs,1);
telapsed_cgal2 = cell(no_runs,1);
telapsed_admm  = cell(no_runs,1);

for k=1:no_runs
    disp("--------------current iteration------------")
    k

    A = randn(s1,n)/n;
    A = A+A';
    A = A + eye(n) * 50/n;
   
    r = ceil(sqrt(2*n));

    B = randn(s1,n,r);
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

    Primitive1 = @(x) -A*x;
    Primitive2 = @(y,x) y.*x;
    Primitive3 = @(x) sum(x.^2,2);
    a = n;
    b = ones(n,1);
    
    % Compute scaling factors
    SCALE_X = 1/n;
    SCALE_C = 1/norm(A,'fro');

    K = inf;
    beta0 = 1;
    maxit = 1e6; % limit on number of iterations
    
    timer = tic;
    cputimeBegin = cputime;

    R = 10;
    [out1] = CGAL( n, Primitive1, Primitive2, Primitive3, a, b, R, maxit, beta0, K, ...
        'FLAG_MULTRANK_P1',true,... % This flag informs that Primitive1 can be applied to find AUU' for any size U. 
        'FLAG_MULTRANK_P3',true,... % This flag informs that Primitive3 can be applied to find (A'y)U for any size U.
        'SCALE_X',SCALE_X,... % SCALE_X prescales the primal variable X of the problem
        'SCALE_C',SCALE_C,... % SCALE_C prescales the cost matrix C of the problem
        "WALLTIME",maxTime * 1,... % asdasd
        'stoptol',1e-8); % algorithm stops when 1e-1 relative accuracy is achieved
    
    f_cgal1{k} = -out1.info.feasible_obj';
    telapsed_cgal1{k} = out1.time_real';
    R = r;
    [out2] = CGAL( n, Primitive1, Primitive2, Primitive3, a, b, R, maxit, beta0, K, ...
        'FLAG_MULTRANK_P1',true,... % This flag informs that Primitive1 can be applied to find AUU' for any size U. 
        'FLAG_MULTRANK_P3',true,... % This flag informs that Primitive3 can be applied to find (A'y)U for any size U.
        'SCALE_X',SCALE_X,... % SCALE_X prescales the primal variable X of the problem
        'SCALE_C',SCALE_C,... % SCALE_C prescales the cost matrix C of the problem
        "WALLTIME",maxTime * 1,... % asdasd
        'stoptol',1e-8); % algorithm stops when 1e-1 relative accuracy is achieved

    f_cgal2{k} = -out2.info.feasible_obj';
    telapsed_cgal2{k} = out2.time_real';

    [cost_admm, grad_admm, telapsed_admm{k}, ~] = ADMM_BM(-A,B,maxTime);
    f_admm{k} = -cost_admm;

    eps = 0.5;
    [f_cg2{k}, grad_cg2{k}, telapsed_cg2{k}, ~] = cg_BM2(A, B, maxTime, opts, eps);

    [f_bcm2{k}, grad_bcm2{k}, telapsed_bcm2{k}, ~] = BCM2(A, B, maxTime,opts,eps);
    % size(f_bcm2{k})
    % telapsed_bcm2{k}
end

%%

if n >= 5000 && no_runs >= 10
    writecell(f_bcm2, "f_bcm2.csv")

    writecell(f_bcm,"f_bcm.csv")
    writecell(f_rgd, "f_rgd.csv")
    writecell(f_rtr, "f_rtr.csv")
    writecell(f_cg, "f_cg.csv")
    writecell(f_cg2, "f_cg2.csv")

    writecell(f_cgal1,"f_cgal1.csv")
    writecell(f_cgal2, "f_cgal2.csv")
    writecell(f_admm, "f_admm.csv")

    writecell(telapsed_bcm2, "telapsed_bcm2.csv")

    writecell(telapsed_bcm, "telapsed_bcm.csv")
    writecell(telapsed_rgd, "telapsed_rgd.csv")
    writecell(telapsed_rtr, "telapsed_rtr.csv")
    writecell(telapsed_cg, "telapsed_cg.csv")
    writecell(telapsed_cg2, "telapsed_cg2.csv")

    writecell(telapsed_cgal1, "telapsed_cgal1.csv")
    writecell(telapsed_cgal2, "telapsed_cgal2.csv")
    writecell(telapsed_admm, "telapsed_admm.csv")
end


