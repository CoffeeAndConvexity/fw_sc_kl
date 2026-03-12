function [x, xcost, info] = rgd(A, x0, opts)
    [r, n] = size(x0);
    
    manifold = obliquefactory(r,n);
    problem.M = manifold;
    
    problem.cost  = @(x) -trace(x*A*x');
    problem.egrad = @(x) -2*(x*A);
    problem.ehess = @(x, u) -2*(u*A);

    [x, xcost, info] = steepestdescent(problem, x0, opts); 
    
end

