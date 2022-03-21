    clear; clc; close all;
%% B.i
n=50;

%B.i.a
A = randn(n);
[U,~,~] = svd(A);

%B.i.b
l_max = [1,2,10,30,50,70,100,300,500,700,1000];
l_min = 1;
k_frac_els=zeros(1,size(l_max,2));
k_frac_bls=zeros(1,size(l_max,2));
for i=1:size(l_max,2)
    z = l_min + (l_max(i) - l_min) * rand(n - 2, 1);
    eig_v = [l_min ; l_max(i) ; z];
    L = diag(eig_v);

    %B.ii
    CN = l_max(i)/l_min;

    P = U*L*U';
%     chol(P); %Check that P is positive definite
    q = randn(n,1);
    f = @(x) 1/2*x'*P*x + q'*x;
    
    %B.iii
    grad_f = @(x) P*x + q;
    %Closed-Form method
    x_star_cf = -P\q;
    p_star_cf = f(x_star_cf);

    %B.iv Gradient Descent
    epsilon = 10^-5; %Stopping Criterion
    x_0 = randn(n,1);
    %Exact Line Search
    Expected_Iter = CN*log(f((x_0)-p_star_cf)/epsilon);
    k_els = 0;
    tic;
    x_k = x_0;
    while norm(grad_f(x_k)) >= epsilon
        t_k = norm(grad_f(x_k))^2/(grad_f(x_k)'*P*grad_f(x_k));
        x_k = x_k - t_k*grad_f(x_k);
        k_els = k_els + 1;
    end
    k_frac_els(i)=k_els/Expected_Iter;
    %Backtracking Line Search
    a = 0.4;
    b = 0.7;
    Expected_Iter_BLS = log(f((x_0)-p_star_cf)/epsilon)/log(1/(1-( min(2*l_min*a, 2*b*a*l_min/l_max(i)) )));
    k_bls = 0;
    x_k = x_0;
    while norm(grad_f(x_k)) >= epsilon
        t_k = 1;
        while f(x_k-t_k*grad_f(x_k)) > f(x_k) - a*t_k*grad_f(x_k)'*grad_f(x_k)
            t_k = b*t_k;
        end
        x_k = x_k - t_k*grad_f(x_k); 
        k_bls = k_bls + 1;
    end
    k_frac_bls(i)=k_bls/Expected_Iter_BLS;
end
plot(l_max,k_frac_els,'.-');
hold on;
plot(l_max,k_frac_bls,'.-');
title('$Comparison\ of\ \frac{k}{k_\epsilon}\ of Exact\ and\ Backtracking\ algorthms for\ different\ K$','Interpreter','latex','fontSize',18);
legend({'$Exact Line Search$','$Backtracking Line Search$'},'Interpreter','latex','fontSize',18);
xlabel('$K$','Interpreter','latex','fontSize',18);
ylabel('$\frac{k}{k_\epsilon}$','Interpreter','latex','fontSize',18);