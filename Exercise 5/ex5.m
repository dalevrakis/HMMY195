clear; close all; clc;

p = 1;
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
t = 1;
mu = 10;
f_t = @(x,t) t*c'*x - sum(log(x));
grad_ft = @(x,t) t*c - 1./x;
hessian_ft = @(x)diag(1./(x.^2));

alpha = 0.1;
beta = 0.7;
threshold_1=10^-1;
threshold_2=10^-5;

x_k = x_feas;
inner_iter = 0;
x_vals_FromCvxFeas = x_feas;
% f_vals_FromCvxFeas = f_t(x_feas,t);
tic;
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

        x_vals_FromCvxFeas = [x_vals_FromCvxFeas x_k];
%         f_vals_FromCvxFeas = [f_vals_FromFoundFeas f_t(x_k,t)];
        inner_iter = inner_iter+1;

    end
    
    if(p/t < threshold_2)
        break;
    end
    t=mu*t;
end
tEnd=toc;
x_opt_fromCvxFeas = x_k;

% 2.b find feasible point
x_init = pinv(A)*b;

A_s = [A zeros(p,1)];

t = 1;
h = 10^-4;

x_k = x_init;
s_k = max(-x_init)+h;
inner_iter = 0;
tic;
while(1)
    while(1)
        g_fts = grad_fts(x_k, s_k, t);
        h_fs = hessian_fs(x_k, s_k) + h;

        w = -( A_s * h_fs^-1 * A_s' )^-1 * A_s * h_fs^-1 * g_fts;
        d = -(h_fs)^-1 * (g_fts + A_s' * w);
        
        lambda2=d' * h_fs * d;
        if(lambda2/2 <= threshold_1)
            break;
        end
        
        if(s_k<0)
            break;
        end
        
        dx = d(1:n);
        ds = d(n+1);
        
        t_k = 1;
        while min(x_k + t_k*dx + s_k + t_k*ds) < 0 
            t_k = beta*t_k;
        end

        while f_ts(x_k+t_k*dx , s_k+t_k*ds, t) > f_ts(x_k,s_k,t) + alpha*t_k*g_fts'*d
            t_k = beta*t_k;
        end
        x_k = x_k + t_k*dx;
        s_k = s_k + t_k*ds;
        
        inner_iter = inner_iter+1;
    end
    if(s_k<0)
        break;
    end
    if(p/t < threshold_2)
        break;
    end
    t=mu*t;
end
tEnd=toc;
x_feas_found = x_k;

t = 1;
x_k = x_feas_found;
inner_iter = 0;
x_vals_FromFoundFeas = x_feas;
% f_vals_FromFoundFeas = f_t(x_feas,t);
tic;
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

        x_vals_FromFoundFeas = [x_vals_FromFoundFeas x_k];
%         f_vals_FromFoundFeas = [f_vals_FromFoundFeas f_t(x_k,t)];
        inner_iter = inner_iter+1;

    end
    
    if(p/t < threshold_2)
        break;
    end
    t=mu*t;
end
tEnd=toc;
x_opt_fromFoundFeas = x_k;

% Dual-Primal Algorithn using interior point mehtod
t = 1;
x_0 = ones(n,1);
x_k = x_0;
v_k = ones(p,1);
k = 0;
x_vals_pd = x_0;
% f_vals_tpd = f(x_0);
r = @(x,v,t) [grad_ft(x,t)+A'*v ; A*x-b];

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

        x_vals_pd = [x_vals_pd x_k];
%         f_vals_tpd = [f_vals_tpd f(x_k)];
        k = k+1;

        if norm(r(x_k, v_k, t))<= threshold_1
            break;
        end
    end
    
    if(p/t < threshold_2)
        break;
    end
    t=mu*t;
end
x_opt_pd = x_k;

nz_cvxFeas = non_zero_elements(x_opt_fromCvxFeas);
nz_foundFeas = non_zero_elements(x_opt_fromFoundFeas);
nz_pd = non_zero_elements(x_opt_pd);


if(p == 1)
    x1 = -2 : 0.001 : 2;
    plot(x1, (b-A(1)*x1)/A(2) );
    hold on;
    plotv(x_cvx);
    plot(x_vals_FromCvxFeas(1,:),x_vals_FromCvxFeas(2,:),'-o');
    plot(x_vals_FromFoundFeas(1,:),x_vals_FromFoundFeas(2,:),'-*');
    plot(x_vals_pd(1,:),x_vals_pd(2,:),'-x');
    title('$x_k$ trajectories of different algorithms','Interpreter','latex','fontSize',18);
    xlabel('$x_1$','Interpreter','latex','fontSize',18);
    ylabel('$x_2$','Interpreter','latex','fontSize',18);
    legend({'Affine Equality Constraint','Cvx Optimal Value','Interior Point w/ Cvx feasible point','Interior Point w/ log barrier feasible point','Interior Point w/ dual-primal algorithm from (1,1)'},'Interpreter','latex','fontSize',18)
%     xlim([-1,2]);
%     ylim([-1,2]);
end

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
end