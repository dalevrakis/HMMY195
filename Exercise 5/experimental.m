clear; close all; clc;

p = 3;
n = 2*p;

% Construct A and b
A = randn(p,n);
% A = [-0.990532220484176,-1.72542778952869,-1.59418372026681,0.787066676357980;-1.17303232726741,0.288228089665011,0.110218849223381,-0.00222678631383613];
x_s = rand(n,1);
b = A*x_s;
% b = [-1.11048654848771;-0.487629913987253];

%Construct cost vector c
z = randn(p,1);
s = rand(n,1);
c = A'*z + s;
% c = [-2.72045149679617;-0.948362146067767;-0.775164983606318;1.39267463105631];

% 1. CVX solution
cvx_begin
    variable x(n,1)
    minimize( c'*x );
    subject to
        A*x == b;
        x >= 0;
cvx_end
fprintf('Search Time: %e secs\n\n',cvx_cputime);

x_cvx = cvx_optpnt.x;

% 2. a CVX feasible point
cvx_begin
    variable x(n,1)
    minimize( 0 );
    subject to
        A*x == b;
        x >= 0;
cvx_end
fprintf('Search Time: %e secs\n\n',cvx_cputime);

x_feas = cvx_optpnt.x;

%Equiv Problem: f_t(x) = t*c'*x - sum_i log(x_i)
mu = 2 : 2 : 200;
f_t = @(x,t) t*c'*x - sum(log(x));
grad_ft = @(x,t) t*c - 1./x;
hessian_ft = @(x)diag(1./(x.^2)) + 10^-8*eye(n);

alpha = 0.1;
beta = 0.7;
threshold_1=10^-1;
threshold_2=10^-5;

h = 10^-4;
newton_iters = zeros(size(mu));
% x_vals_FromCvxFeas = x_feas;
% f_vals_FromCvxFeas = f_t(x_feas,t);
tic;
for i = 1 : size(mu,2)
    t = 1;
    x_k = x_feas;
    inner_iter = 0;
    while(1)
        while(1)
            g_ft = grad_ft(x_k,t);
            h_ft = hessian_ft(x_k);

            w = -inv( A/h_ft*A' )*A/h_ft*g_ft;
            dx = -inv(h_ft)*(g_ft+A'*w);

            lambda2=dx'*h_ft*dx;
    %         disp(lambda2/2);
            if(lambda2/2 <= threshold_1)
                break;
            end

            t_k = 1;   
            while min(x_k+t_k*dx) < 0       
                t_k = beta*t_k;
            end

            while f_t(x_k+t_k*dx,t) > f_t(x_k,t) + alpha*t_k*g_ft'*dx
                t_k = beta*t_k;
            end
            x_k = x_k + t_k*dx;

    %         x_vals_FromCvxFeas = [x_vals_FromCvxFeas x_k];
    %         f_vals_FromCvxFeas = [f_vals_FromFoundFeas f_t(x_k,t)];
            inner_iter = inner_iter+1;

        end

        if(p/t < threshold_2)
            break;
        end
        t=mu(i)*t;
    end
    newton_iters(i) = inner_iter;
end
tEnd=toc;
x_opt_fromCvxFeas = x_k;

figure;
plot(mu,newton_iters,'-x')
hold on;
plot(mu,mean(newton_iters)*ones(size(mu)));
xlabel('$\mu$','Interpreter','latex','fontSize',18);
ylabel('Newton Iterations','Interpreter','latex','fontSize',18);
legend({'Newton Iterations','Mean Newton Iterations'},'Interpreter','latex','fontSize',18);
hold off;

t = 1;
x_0 = ones(n,1);
x_k = x_0;
v_k = ones(p,1);
k = 0;
% x_vals_pd = x_0;
% f_vals_tpd = f(x_0);
r = @(x,v,t) [grad_ft(x,t)+A'*v ; A*x-b];

iters_pd = zeros(size(mu));
for i = 1 : size(mu,2)
    t = 1;
    x_k = x_feas;
    inner_iter = 0;
    while(1)
        while(1)
            g_f = grad_ft(x_k,t);
            h_f = hessian_ft(x_k);

            dv = -inv( A/h_f*A')*(-(A*x_k-b)+A/h_f*(g_f+A'*v_k));
            dx = -inv(h_f)*(g_f+A'*v_k+A'*dv);

            t_k = 1;
            while min(x_k+t_k*dx) < 0       
                t_k = beta*t_k;
            end

            while norm(r(x_k+t_k*dx, v_k+t_k*dv, t)) > (1-alpha*t_k)*norm(r(x_k, v_k, t))
                t_k = beta*t_k;
            end
            x_k = x_k + t_k*dx;
            v_k = v_k + t_k*dv;

%             x_vals_pd = [x_vals_pd x_k];
    %         f_vals_tpd = [f_vals_tpd f(x_k)];
            inner_iter = inner_iter+1;

            if norm(r(x_k, v_k, t))<= threshold_1
                break;
            end
        end

        if(p/t < threshold_2)
            break;
        end
        t=mu(1)*t;
    end
    
    iters_pd(i) = inner_iter;
end
x_opt_pd = x_k;

figure;
plot(mu,iters_pd,'-x')
hold on;
plot(mu,mean(iters_pd)*ones(size(mu)));
xlabel('$\mu$','Interpreter','latex','fontSize',18);
ylabel('Newton Iterations','Interpreter','latex','fontSize',18);
legend({'Newton Iterations','Mean Newton Iterations'},'Interpreter','latex','fontSize',18);
hold off;

%Calculate non zero elements of optimal points found by each algorithm
nz_cvxFeas = non_zero_elements(x_opt_fromCvxFeas);
nz_pd = non_zero_elements(x_opt_pd);

function c = non_zero_elements(x)
    c = 0;
    for i = 1:size(x,1)
        if (abs(x(i)) < 10^-3)
            c = c + 1;
        end 
    end
end
function f = f_ts(x,s,t)
    n = size(x,1);
    
    f = t*s;
    for i=1:n
        f = f - log(x(i)+s);
    end
end

function g = grad_fts(x,s,t)
    n = size(x,1);
    I = eye(n+1);

    g = t*I(:,n+1);

    for i=1:n
        g = g - ( 1/(x(i)+s) ) * ( I(:,i)+I(:,n+1) );
    end
end

function h = hessian_fs(x,s)
    n = size(x,1);
    I = eye(n+1);

    h = 0;
    for i=1:n
        h = h + (1/(x(i)+s)^2)*(I(:,i)+I(:,n+1))*(I(:,i)+I(:,n+1))';
    end
    h = h + 10^-8*eye(n+1);
end