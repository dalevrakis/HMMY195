clear; clc; close all;
%% B.i
n=2;

%B.i.a
A = randn(n);
[U,~,~] = svd(A);

disp(U*U'); %Almost equal to I
disp(U'*U);

%B.i.b
l_max = [1,2,10, 100, 1000];
l_min = 1;
fprintf('*************************  n=%d  *************************\n',n);
for i=1:size(l_max,2)
    z = l_min + (l_max(i) - l_min) * rand(n - 2, 1);
    eig_v = [l_min ; l_max(i) ; z];
    L = diag(eig_v);

    %B.ii
    CN = l_max(i)/l_min;
    fprintf('--------------------------------------------------------\n');
    fprintf('Condition Number = %.0f\n',CN);

    P = U*L*U';
%     chol(P); %Check that P is positive definite
    q = randn(n,1);
    f = @(x) 1/2*x'*P*x + q'*x;
    f2 = @(x1,x2) 1/2*(x1.^2.*P(1,1)+x1.*x2.*P(2,1)+x1.*x2.*P(1,2)+x2.^2.*P(2,2))+q(1).*x1+q(2).*x2;

    %B.iii
    grad_f = @(x) P*x + q;
    %Closed-Form method
    x_star_cf = -P\q;
    p_star_cf = f(x_star_cf);
%     grad_f(x_star_cf) %Check that grad_f(x_star_cf) is indeed equal to 0;
    fprintf('Closed-Form Solution: p* = %f',p_star_cf);
    fprintf('\n\n');

    %B.iv Gradient Descent
    epsilon = 10^-5; %Stopping Criterion
    x_0 = randn(n,1);
    %Exact Line Search
    Expected_Iter = CN*log(f((x_0)-p_star_cf)/epsilon);
    k_els = 0;
    tic;
    x_k = x_0;
    x_vals_els = x_0;
    while norm(grad_f(x_k)) >= epsilon
        t_k = norm(grad_f(x_k))^2/(grad_f(x_k)'*P*grad_f(x_k));
        x_k = x_k - t_k*grad_f(x_k);
        x_vals_els = [x_vals_els x_k];
        k_els = k_els + 1;
    end
    tEnd=toc;
    fprintf('Exact Line Search (Maximum Expected Iterations:%d):\n',ceil(Expected_Iter));
    fprintf('p* = %f\n',f(x_k));
    fprintf('x* error MSE = %e\n',mse(x_star_cf, x_k));
    fprintf('Iterations: %d\n',k_els);
    fprintf('Search Time: %.e secs\n',tEnd);
    fprintf('\n');

    %Backtracking Line Search
    a = [0.1, 0.2, 0.3, 0.4];
    b = [0.1, 0.4, 0.7, 0.9];
    k_bls_min = 10^5;
    k_bls_max = 0;
    figure;
    for alpha=1:size(a,2)
        for beta=1:size(b,2)
            Expected_Iter_BLS = log(f((x_0)-p_star_cf)/epsilon)/log(1/(1-( min(2*l_min*a(alpha), 2*b(beta)*a(alpha)*l_min/l_max(i)) )));
            k_bls = 0;
            x_k = x_0;
            tic;
            x_vals_bls = x_0;
            while norm(grad_f(x_k)) >= epsilon
                t_k = 1;
                while f(x_k-t_k*grad_f(x_k)) > f(x_k) - a(alpha)*t_k*grad_f(x_k)'*grad_f(x_k)
                    t_k = b(beta)*t_k;
                end
                x_k = x_k - t_k*grad_f(x_k); 
                x_vals_bls = [x_vals_bls x_k];
                k_bls = k_bls + 1;
            end
            tEnd = toc;
            fprintf('Backtracking Search(a=%.2f,b=%.2f, Maximum Expected Iterations:%d):\n',a(alpha),b(beta),ceil(Expected_Iter_BLS));
            fprintf('p* = %f\n',f(x_k));
            fprintf('x* error MSE = %e\n',mse(x_star_cf, x_k));
            fprintf('Iterations: %d\n',k_bls);
            fprintf('Search Time: %.e secs\n',tEnd);
            fprintf('--------------------------------------------------------');

            fprintf('\n');
            f_vals_bls = f2(x_vals_bls(1,:),x_vals_bls(2,:));
            if(k_bls > k_bls_max)
                a_max = a(alpha);
                b_max = b(beta);
                x_vals_bls_max = x_vals_bls;
                f_vals_bls_max = f_vals_bls;
                k_bls_max = k_bls;
            elseif(k_bls < k_bls_min )
                a_min = a(alpha);
                b_min = b(beta);
                x_vals_bls_min = x_vals_bls;
                f_vals_bls_min = f_vals_bls;
                k_bls_min = k_bls;
            end
            
            %B.vi
            plot(0:k_bls,log(f_vals_bls-p_star_cf));
            hold on;
        end
    end
    hold off
    ylabel('$f(x)-p^{*}$','Interpreter','latex','fontSize',18);
    xlabel('$k$','Interpreter','latex','fontSize',18);
    title('$Comparison\ of\ Backtracking\ Search\ Value\ Error\ to\ Iteration$','Interpreter','latex','fontSize',18);
    legend({'$a=0.1,\ b=0.1$','$a=0.1,\ b=0.4$','$a=0.1, b=0.7$','$a=0.1, b=0.9$','$a=0.2, b=0.1$','$a=0.2, b=0.4$','$a=0.2, b=0.7$','$a=0.2, b=0.9$','$a=0.3, b=0.1$','$a=0.3, b=0.4$','$a=0.3, b=0.7$','$a=0.3, b=0.9$','$a=0.4, b=0.1$','$a=0.4, b=0.4$','$a=0.4, b=0.7$','$a=0.4, b=0.9$'},'Interpreter','latex','fontSize',18);

    %B.v
    f = @(x1,x2) 1/2*(x1.^2.*P(1,1)+x1.*x2.*P(2,1)+x1.*x2.*P(1,2)+x2.^2.*P(2,2))+q(1).*x1+q(2).*x2;

    % Backtracking Line Search Plots
    min_bls_x = min(x_vals_bls(1,:));
    max_bls_x = max(x_vals_bls(1,:));
    min_bls_y = min(x_vals_bls(2,:));
    max_bls_y = max(x_vals_bls(2,:));
    [x_1, x_2] = meshgrid(min_bls_x-0.1 : (max_bls_x-min_bls_x)/1000 : max_bls_x+0.1 , min_bls_y-0.1 : (max_bls_y-min_bls_y)/100 : max_bls_y+0.1);
    figure;
    contour(x_1,x_2,f2(x_1,x_2),f_vals_bls_max);
    hold on;
    plot(x_vals_bls_max(1,:),x_vals_bls_max(2,:),'r.-');
    contour(x_1,x_2,f2(x_1,x_2),f_vals_bls_min);
    plot(x_vals_bls_min(1,:),x_vals_bls_min(2,:),'b.-');
    xlim([min_bls_x-0.1, max_bls_x+0.1]);
    ylim([min_bls_y-0.1, max_bls_y+0.1])
    hold off;
    title('$Comparison\ of\ Fastes\ and\ Slowest\ Parameters\ of\ Backtracking\ Line\ Search$','Interpreter','latex','fontSize',18);
    fast = sprintf('$a=%.1f, b=%.1f$',a_min,b_min);
    slow = sprintf('$a=%.1f, b=%.1f$',a_max,b_max);
    legend({'',fast,'',slow},'Interpreter','latex','fontSize',18);
    xlabel('$x_1$','Interpreter','latex','fontSize',18);
    ylabel('$x_2$','Interpreter','latex','fontSize',18);

    % Exact Line Search Plots
    f_vals_els = f2(x_vals_els(1,:),x_vals_els(2,:));
    
    min_els_x = min(x_vals_els(1,:));
    max_els_x = max(x_vals_els(1,:));
    min_els_y = min(x_vals_els(2,:));
    max_els_y = max(x_vals_els(2,:));
    [x_1, x_2] = meshgrid(min_els_x-0.1 : (max_els_x-min_els_x)/1000 : max_els_x+0.1 , min_els_y-0.1 : (max_els_y-min_els_y)/100 : max_els_y+0.1);
    figure;
    contour(x_1,x_2,f2(x_1,x_2),f_vals_els);
    hold on;
    plot(x_vals_els(1,:),x_vals_els(2,:),'.-');
    xlim([min_els_x-0.1, max_els_x+0.1]);
    ylim([min_els_y-0.1, max_els_y+0.1])
    hold off;
    title('$Exact\ Line\ Search$','Interpreter','latex','fontSize',18);
    xlabel('$x_1$','Interpreter','latex','fontSize',18);
    ylabel('$x_2$','Interpreter','latex','fontSize',18);

    %B.vi
    figure;
    plot(0:k_els,log(f_vals_els-p_star_cf),'.-');
    title('$Exact\ Line\ Search\ Value\ Error\ to\ Iteration$','Interpreter','latex','fontSize',18)
    ylabel('$f(x)-p^{*}$','Interpreter','latex','fontSize',18);
    xlabel('$k$','Interpreter','latex','fontSize',18);
end