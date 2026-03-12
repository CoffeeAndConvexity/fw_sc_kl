clear all
close all
clc

% rng(0,"twister");
% add CGAL codes, note that we had to change the code a little bit to
% compute feasible objective and enforce REAL WALLTIME, the code itself
% uses totCpuTime but control message says "walltime achieved" not to
% mention it also burdens itself with in between logging computations that
% are not solver related in its original form. 
addpath utils;
addpath solver;

s1 = RandStream('twister','Seed',0);

% n = 500;
n = 20000;

maxNumCompThreads(16)
r = ceil(sqrt(2*n));
% 
% maxTime = 4; 
% no_runs = 1;

maxTime = 60; 
no_runs = 50;

f_cgal1 = cell(no_runs,1);
f_cgal2 = cell(no_runs,1);
f_admm  = cell(no_runs,1);


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
    
    f_cgal1{k} = -out1.info.feasible_obj;
    telapsed_cgal1{k} = out1.time_real;
    R = r;
    [out2] = CGAL( n, Primitive1, Primitive2, Primitive3, a, b, R, maxit, beta0, K, ...
        'FLAG_MULTRANK_P1',true,... % This flag informs that Primitive1 can be applied to find AUU' for any size U. 
        'FLAG_MULTRANK_P3',true,... % This flag informs that Primitive3 can be applied to find (A'y)U for any size U.
        'SCALE_X',SCALE_X,... % SCALE_X prescales the primal variable X of the problem
        'SCALE_C',SCALE_C,... % SCALE_C prescales the cost matrix C of the problem
        "WALLTIME",maxTime * 1,... % asdasd
        'stoptol',1e-8); % algorithm stops when 1e-1 relative accuracy is achieved

    f_cgal2{k} = -out2.info.feasible_obj;
    telapsed_cgal2{k} = out2.time_real;

    [cost_admm, grad_admm, telapsed_admm{k}, B] = ADMM_BM(-A,B,maxTime);
    f_admm{k} = -cost_admm;

end

%%
% set(groot,'defaultLineLineWidth',.5);
% figure;
% set(gca,'fontsize',24);
% hold on;
% for k=1:no_runs
%     semilogy(telapsed_rtr{k}, f_rtr{k}, '-r');
%     semilogy(out.time,-1 * out.info.primalObj,"-k")
% end
% 
% xlabel('Time (s)', 'FontSize', 30, 'interpreter','latex');
% ylabel('$f(B^k)$', 'FontSize', 30, 'interpreter','latex');
% title(['n = '  num2str(n) ' and r = ' num2str(r)]);
% legend("BCM","RGD","RTR","CG","CG2",'Location','southeast');
% xlim([0, maxTime]);

%%
% 
% G = load("random_data/A10.mat");
% norm(A-G.A,"fro")

if n >= 2000 && no_runs >= 5
    writecell(f_cgal1,"f_cgal1.csv")
    writecell(f_cgal2, "f_cgal2.csv")
    writecell(f_admm, "f_admm.csv")

    writecell(telapsed_cgal1, "telapsed_cgal1.csv")
    writecell(telapsed_cgal2, "telapsed_cgal2.csv")
    writecell(telapsed_admm, "telapsed_admm.csv")
end


