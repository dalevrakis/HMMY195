clear; clc; close all;

% 1. Data Generation
p = 1;
n = 2;

A = rand(p,n);
x_s = rand(n,1);

b = A*x_s;

f=@(x) -sum(log(x));
f2=@(x1,x2) -log(x1)-log(x2);
% 2. CVX solution
cvx_begin
    variable x(n,1)
    minimize( - sum(log(x)) );
    subject to
        A*x == b;
cvx_end
fprintf('Search Time: %e secs\n\n',cvx_cputime);

x_cvx = cvx_optpnt.x;

% 3. Newton solution

% 3.i. Compute a feasible point via cvx
cvx_begin
    variable x(n,1)
    minimize( 0 );
    subject to
        A*x==b;
        x > 0;
cvx_end
x_0 = cvx_optpnt.x;

% 3.ii. Compute solution with Newton algorithm starting from x0
grad_f = @(x) -1./x;
hessian_f = @(x) diag(1./(x.^2));

alpha = 0.25;
beta = 0.7;
x_k = x_0;
k_newt = 0;
x_vals_newt = x_0;
f_vals_newt = f(x_0);
epsilon=10^-10;
tic;
while(1)
    g_f = grad_f(x_k);
    h_f = hessian_f(x_k);
    
    w = -inv( A/h_f*A' )*A/h_f*g_f;
    dx = -inv(h_f)*(g_f+A'*w);
    
    t_k = 1;   
    while min(x_k+t_k*dx) < 0       
        t_k = beta*t_k;
    end
    
    while f(x_k+t_k*dx) > f(x_k) + alpha*t_k*g_f'*dx
        t_k = beta*t_k;
    end
    x_k = x_k + t_k*dx;
    
    x_vals_newt = [x_vals_newt x_k];
    f_vals_newt = [f_vals_newt f(x_k)];
    k_newt = k_newt+1;
    
    lambda2=dx'*h_f*dx;
    if(lambda2/2 <= epsilon)
        break;
    end
end
tEnd=toc;
fprintf("Newton descend from feasible point run time:%d\n",tEnd);
min_newt_x = min(x_vals_newt(1,:));
max_newt_x = max(x_vals_newt(1,:));
min_newt_y = min(x_vals_newt(2,:));
max_newt_y = max(x_vals_newt(2,:));
[x_1, x_2] = meshgrid(min_newt_x-0.1 : (max_newt_x-min_newt_x)/100 : max_newt_x+0.1 , min_newt_y-0.1 : (max_newt_y-min_newt_y)/100 : max_newt_y+0.1);

figure;
contour(x_1,x_2,f2(x_1,x_2),f_vals_newt);
hold on;
plot(x_vals_newt(1,:),x_vals_newt(2,:),'-o');
plot(x_1(1,:),(b-A(1)*x_1(1,:))/A(2));
hold off;
legend({'$f\ level\ Sets$','$x_k\ iterations$','$Ax=b$'},'Interpreter','latex','fontSize',18);
xlim([min_newt_x-0.1 max_newt_x+0.1]);
ylim([min_newt_y-0.1 max_newt_y+0.1]);

% 3.iii Plot the norms of x_k in each iteration against the solution from
% cvx
norm_xk = zeros(1,size(x_vals_newt,2));
norm_x_cvx = zeros(1,size(x_vals_newt,2));
for i=1:k_newt+1
    norm_xk(i)=norm(x_vals_newt(:,i));
    norm_x_cvx(i)=norm(x_cvx);
end

figure;
plot(0:k_newt,norm_x_cvx,'--');
hold on; 
plot(0:k_newt,norm_xk,'--*');
hold off;
title('$Plot\ x_k\ norms\ in\ each\ iteration\ against\ cvx\ norm\ solution$','Interpreter','latex','fontSize',18);
xlabel('$k$','Interpreter','latex','fontSize',18);
legend({'$||x_{cvx}||_2$','$||x_k||_2$'},'Interpreter','latex','fontSize',18);
pause(0.1);

% 4.
x_0 = ones(n,1);
x_k = x_0;
v_k = ones(p,1);
k_newtpd = 0;
x_vals_newtpd = x_0;
f_vals_newtpd = f(x_0);
r = @(x,v) [grad_f(x)+A'*v ; A*x-b];
tic;
while(1)
    g_f = grad_f(x_k);
    h_f = hessian_f(x_k);
    
    dv = -inv( A/h_f*A')*(-(A*x_k-b)+A/h_f*(g_f+A'*v_k));
    dx = -inv(h_f)*(g_f+A'*v_k+A'*dv);
    
    t_k = 1;
    while min(x_k+t_k*dx) < 0       
        t_k = beta*t_k;
    end
    
    while norm(r(x_k+t_k*dx,v_k+t_k*dv)) > (1-alpha*t_k)*norm(r(x_k, v_k))
        t_k = beta*t_k;
    end
    x_k = x_k + t_k*dx;
    v_k = v_k + t_k*dv;
    
    x_vals_newtpd = [x_vals_newtpd x_k];
    f_vals_newtpd = [f_vals_newtpd f(x_k)];
    k_newtpd = k_newtpd+1;
    
    if norm(r(x_k,v_k))<= epsilon
        break;
    end
end
tend=toc;
fprintf("Dual primal newton descend from any point run time:%d\n",tEnd);

min_newtpd_x = min(x_vals_newtpd(1,:));
max_newtpd_x = max(x_vals_newtpd(1,:));
min_newtpd_y = min(x_vals_newtpd(2,:));
max_newtpd_y = max(x_vals_newtpd(2,:));
[x_1, x_2] = meshgrid(min_newtpd_x-0.1 : (max_newtpd_x-min_newtpd_x)/100 : max_newtpd_x+0.1 , min_newtpd_y-0.1 : (max_newtpd_y-min_newtpd_y)/100 : max_newtpd_y+0.1);

figure;
contour(x_1,x_2,f2(x_1,x_2),f_vals_newtpd);
hold on;
plot(x_vals_newtpd(1,:),x_vals_newtpd(2,:),'-o');
plot(x_1(1,:),(b-A(1)*x_1(1,:))/A(2));
hold off;
legend({'$f\ level\ Sets$','$x_k\ iterations$','$Ax=b$'},'Interpreter','latex','fontSize',18);
xlim([min_newtpd_x-0.1 max_newtpd_x+0.1]);
ylim([min_newtpd_y-0.1 max_newtpd_y+0.1]);

norm_xkpd = zeros(1,size(x_vals_newtpd,2));
norm_x_cvxpd = zeros(1,size(x_vals_newtpd,2));

for i=1:k_newtpd+1
    norm_xkpd(i)=norm(x_vals_newtpd(:,i));
    norm_x_cvxpd(i)=norm(x_cvx);
end

figure;
plot(0:k_newtpd,norm_x_cvxpd,'--');
hold on; 
plot(0:k_newtpd,norm_xkpd,'--*');
hold off;
title('$Plot\ x_k\ norms\ in\ each\ iteration\ against\ cvx\ norm\ solution$','Interpreter','latex','fontSize',18);
xlabel('$k$','Interpreter','latex','fontSize',18);
legend({'$||x_{cvx}||_2$','$||x_k||_2$'},'Interpreter','latex','fontSize',18);

% 5.
cvx_begin
    variable v(p,1)
    maximize( -b'*v + n + sum(log(A'*v)) );
cvx_end

v_dual = cvx_optpnt.v;

x_primal = 1./(A'*v_dual);
