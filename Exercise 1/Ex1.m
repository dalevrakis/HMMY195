%% 1
clear; clc; close all;
%function definitions
f = @(x) 1./(1+x);
f_prime = @(x) -1./(1+x).^2; %%f'
f_prime2 =@(x) 2./(1+x).^3; %%f''


f_1 = @(x,x_0) f(x_0) + f_prime(x_0)*(x-x_0); %%First order Taylor aproximation of f at x_0
f_2 = @(x,x_0) f(x_0) + f_prime(x_0)*(x-x_0) + 1/2*f_prime2(x_0)*(x-x_0).^2; %%Second order Taylor aproximation of f at x_0

x = 0:0.1:40; %%x axis
x_0 = 30;
figure;
plot(x,f(x));
hold on;
plot(x,f_1(x,x_0));
plot(x,f_2(x,x_0));
hold off;
ylim([-0.5,1]);
legend({'$f(x)$','$f_1(x)$','$f_2(x)$'},'Interpreter','latex');
fontSize=14;
 

%% 2
clear; clc; close all;
f = @(x_1, x_2) 1./(1+x_1+x_2);
x_star = 25;

%a
figure;
[x_1, x_2] = meshgrid(0:0.05:x_star);
mesh(x_1, x_2, f(x_1,x_2));
fontSize = 18;
title('$Plot\ of\ f(x_1,x_2)=\frac{1}{1+x_1+x_2},x_1,x_2\in[0,x^{*}],x^{*}=25$','Interpreter','latex','fontSize',fontSize);
legend({'$f(x)$','$f_1(x)$','$f_2(x)$'},'Interpreter','latex','fontSize',fontSize);
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);
colorbar;
caxis([0,1]);

%b
figure;
contour(x_1, x_2, f(x_1,x_2));
xlim([0,12]);
ylim([0,12]);
title('$Contour\ of\ f(x_1,x_2)=\frac{1}{1+x_1+x_2},x_1,x_2\in[0,x^{*}],x^{*}=25$','Interpreter','latex','fontSize',fontSize);
legend({'$f(x)$','$f_1(x)$','$f_2(x)$'},'Interpreter','latex','fontSize',fontSize);
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);
colorbar;
caxis([0,1]);
%c
Grad =@(x_1,x_2) [-1./(1+x_1+x_2).^2 ; -1./(1+x_1+x_2).^2];
Hessian =@(x_1,x_2) [ 2./(1+x_1+x_2).^3 , 2./(1+x_1+x_2).^3 ; 2./(1+x_1+x_2).^3 , 2./(1+x_1+x_2).^3];

f_1 = @(x_1,x_2,x_01,x_02)  f(x_01,x_02) + (-1./(1+x_01+x_02).^2)*(x_1-x_01) + (-1./(1+x_01+x_02).^2)*(x_2-x_02);
f_2 = @(x_1,x_2,x_01,x_02) f_1(x_1,x_2,x_01,x_02) +1/2*(x_1-x_01+x_2-x_02).^2*2./(1+x_1+x_2).^3;

x_01=10;
x_02=6;

figure;
mesh(x_1, x_2, f(x_1,x_2),'edgecolor','b');
hold on;
mesh(x_1, x_2, f_1(x_1,x_2,x_01,x_02),'edgecolor','r');
hold off;
xlim([0,13]);
ylim([0,13]);
zlim([0,1]);
legend({'$f(x)$','$f_1(x)$'},'Interpreter','latex','fontSize',fontSize);
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);
title('$$Visualization\ of\ f(x)\ ,First(f_1(x))\ order\ taylor\ aproximations\ at\ (x_{01},x_{02})=(10,6)$$','Interpreter','latex','fontSize',fontSize);


figure;
mesh(x_1, x_2, f(x_1,x_2),'edgecolor','b');
hold on;
mesh(x_1, x_2, f_2(x_1,x_2,x_01,x_02),'edgecolor','g');
hold off;
xlim([0,13]);
ylim([0,13]);
zlim([0,1]);
legend({'$f(x)$','$f_2(x)$'},'Interpreter','latex','fontSize',fontSize);
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);
title('$$Visualization\ of\ f(x)\ ,Second(f_2(x))\ order\ taylor\ aproximations\ at\ (x_{01},x_{02})=(10,6)$$','Interpreter','latex','fontSize',fontSize);

%% 5d
clear; clc; close all;
f=@(x_1,x_2) sqrt(x_1.^2+x_2.^2);
[x_1, x_2] = meshgrid(-50:0.1:50);
mesh(x_1, x_2, f(x_1, x_2));
fontSize=18;
title('$Visualization\ of\ f(x)=||x||_2$','Interpreter','latex','fontSize',fontSize);
%% 6b
clear; clc; close all;
m=3;
n=2;

A=rand(m,n).*10 - 5;
x=rand(n,1).*10 - 5;
b=A*x;

f = @(x_1,x_2) (A(1,1)*x_1+A(1,2)*x_2-b(1)).^2 + (A(2,1)*x_1+A(2,2)*x_2-b(2)).^2 + (A(3,1)*x_1+A(3,2)*x_2-b(3)).^2;

figure;
subplot(3,2,1);
[x_1, x_2] = meshgrid(-100:1:100);
mesh(x_1, x_2, f(x_1, x_2), 'edgecolor','r' );
xlim([x(1)-100,x(1)+100]);
ylim([x(2)-100,x(2)+100]);
fontSize=18;
title('$f(x)$','Interpreter','latex','fontSize',fontSize);
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);

subplot(3,2,2);
contour(x_1,x_2,f(x_1,x_2), 'edgecolor','r');
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);

e=normrnd(5,3);
b_noise=A*x+e;
f_noise = @(x_1,x_2) (A(1,1)*x_1+A(1,2)*x_2-b_noise(1)).^2 + (A(2,1)*x_1+A(2,2)*x_2-b_noise(2)).^2 + (A(3,1)*x_1+A(3,2)*x_2-b_noise(3)).^2;

subplot(3,2,3);
mesh(x_1, x_2, f_noise(x_1, x_2), 'edgecolor','b' );
xlim([x(1)-100,x(1)+100]);
ylim([x(2)-100,x(2)+100]);
title('$f(x)\ with\ small\ error\ e$','Interpreter','latex','fontSize',fontSize);
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);

subplot(3,2,4);
contour(x_1,x_2,f_noise(x_1,x_2), 'edgecolor','b');
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);

subplot(3,2,5);
mesh(x_1, x_2, f(x_1, x_2),'edgecolor','r' );
hold on;
mesh(x_1, x_2, f_noise(x_1, x_2),'edgecolor','b' );
xlim([x(1)-100,x(1)+100]);
ylim([x(2)-100,x(2)+100]);
title('$f(x)\ and\ f(x)\ with\ small\ error\ Overlapped$','Interpreter','latex','fontSize',fontSize);
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);
legend({'f(x)','$f_{noise}(x)$'},'Interpreter','latex','fontSize',fontSize);

subplot(3,2,6);
contour(x_1, x_2, f(x_1, x_2),'edgecolor','r' );
hold on;
contour(x_1, x_2, f_noise(x_1, x_2),'edgecolor','b' );
xlabel('$x_1$','Interpreter','latex','fontSize',fontSize);
ylabel('$x_2$','Interpreter','latex','fontSize',fontSize);
legend({'f(x)','$f_{noise}(x)$'},'Interpreter','latex','fontSize',fontSize);
