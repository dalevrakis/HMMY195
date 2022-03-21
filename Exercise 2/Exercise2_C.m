clear; clc; close all;
%% C

n = [2, 50, 300];
m = [20, 200, 800];

for i = 1 : 3
    fprintf('--------------[n=%d , m=%d]--------------',n(i),m(i));
    A = randn(m(i), n(i));
    b = rand(m(i),1);
    c = randn(n(i),1);
    
    %C.a
    cvx_begin
        variable x(n(i),1)
        minimize( c'*x - sum(log(b-A*x)) );
    cvx_end
    fprintf('Search Time: %e secs\n\n',cvx_cputime);
    
    %C.b
    log_arg =@(x) b - A*x;
    if n(i) == 2
        x1 = cvx_optpnt.x(1)-1 : 2/1000 : cvx_optpnt.x(1)+1;
        x2 = cvx_optpnt.x(2)-1 : 2/1000 : cvx_optpnt.x(2)+1;
        f = zeros(size(x1,2),size(x1,2));
        
        for k = 1:size(x1,2)
            for j = 1:size(x2,2)
                lg_arg = log_arg( [x1(k) ; x2(j)] );
                if min(lg_arg) >= 0
                    f(k,j) = c'*[x1(k);x2(j)] - sum(log(lg_arg));
                else
                    f(k,j) = -inf;
                end
            end
        end
        figure;
        mesh(x1,x2,f);
        title('$f(x)$','Interpreter','latex','fontSize',18);
        xlabel('$x_1$','Interpreter','latex','fontSize',18);
        ylabel('$x_2$','Interpreter','latex','fontSize',18);
        zlabel('$f(x)$','Interpreter','latex','fontSize',18);
        
        figure;
        contour(x1,x2,f);
        title('$f(x)$','Interpreter','latex','fontSize',18);
        xlabel('$x_1$','Interpreter','latex','fontSize',18);
        ylabel('$x_2$','Interpreter','latex','fontSize',18);
        zlabel('$f(x)$','Interpreter','latex','fontSize',18);
    end    
    
    %C.c
    f = @(x) c'*x - sum(log(b-A*x));
    epsilon = 10^-3; %Stopping Criterion
    x_0 = zeros(n(i),1);
    alpha = 0.5;
    beta = 0.7;
    k_grad = 0;
    x_k = x_0;
    
    grad = grad_f(x_k,m(i),A,b,c);
    tic;
    x_vals_grad = x_0;
    while norm(grad) >= epsilon
        t_k = 1;    
        while min(log_arg(x_k-t_k*grad)) < 0
            t_k = beta*t_k;
        end

        while f(x_k-t_k*grad) > f(x_k) - alpha*t_k*grad'*grad
            t_k = beta*t_k;
        end
        x_k = x_k - t_k*grad; 
    
        grad = grad_f(x_k,m(i),A,b,c);
        x_vals_grad = [x_vals_grad x_k];
        k_grad = k_grad + 1;
    end
    tEnd=toc;
    fprintf('Gradient Descend with Bactracking Line Search\n');
    x_opt_grad = x_k;
    optval_grad = f(x_k);
    fprintf('p* = %f\n',optval_grad);
    fprintf('x* error MSE with cvx= %e\n',mse(cvx_optpnt.x, x_k));
    fprintf('p* error MSE with cvx= %e\n',mse(cvx_optval, optval_grad));
    fprintf('Iterations: %d\n',k_grad);
    fprintf('Search Time: %.e secs\n',tEnd);
    fprintf('\n');
    
    %C.d
    k_newt = 0;
    x_k = x_0;
    x_vals_newt = x_0;
    tic;
    while(1)
        hessian = hessian_f(x_k,A,b,n(i),m(i));
        grad = grad_f(x_k,m(i),A,b,c);
        
        lambda2 = grad'/hessian*grad;
        if(lambda2/2 <= epsilon)
            break;
        end
        
        dx = -hessian\grad;
        t_k = 1;   
        while min(log_arg(x_k+t_k*dx)) < 0
            t_k = beta*t_k;
        end

        while f(x_k+t_k*dx) > f(x_k) + alpha*t_k*grad'*dx
            t_k = beta*t_k;
        end
        x_k = x_k + t_k*dx;
        x_vals_newt = [x_vals_newt x_k];
        k_newt = k_newt+1;
    end
    tEnd=toc;
    fprintf('Newton Descend with Bactracking Line Search\n');
    x_opt_newt = x_k;
    optval_newt = f(x_k);
    fprintf('p* = %f\n',optval_newt);
    fprintf('x* error MSE with cvx= %e\n',mse(cvx_optpnt.x, x_k));
    fprintf('p* error MSE with cvx= %e\n',mse(cvx_optval, optval_newt));
    fprintf('Iterations: %d\n',k_newt);
    fprintf('Search Time: %.e secs\n',tEnd);
    fprintf('\n');
    
    %C.e
    figure;
    semilogy(0:k_grad,f(x_vals_grad)-optval_grad);
    hold on;
    semilogy(0:k_newt,f(x_vals_newt)-optval_newt);
    hold off;
    title('$Comparison\ of\ Gradient\ and\ Netwon\ Descend$','Interpreter','latex','fontSize',18);
    xlabel('$k$','Interpreter','latex','fontSize',18);
    ylabel('$f(x_k)-p^*$','Interpreter','latex','fontSize',18);
    legend({'$Gradient$','$Newton$'},'Interpreter','latex','fontSize',18);
end

function h = hessian_f(x,A,b,n,m)
    h = zeros(n,n);
    for i = 1:n
        for j = 1:m
            h(:,i) = h(:,i)+A(j,i)*A(j,:)'/ (b(j)-A(j,:)*x)^2;
        end
    end
            
end
function g = grad_f(x,m,A,b,c)
    g = c;
    for l = 1:m
        g = g + A(l,:)'/(b(l)-(A(l,:)*x));
    end
end 