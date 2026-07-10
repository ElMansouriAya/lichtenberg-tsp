function [g,geq]=constraint(x)
% Inequality constraints
% welded beam design problem
Q=6000*(14+x(2)/2);
D=sqrt(x(2)^2/4+(x(1)+x(3))^2/4);
J=2*(x(1)*x(2)*sqrt(2)*(x(2)^2/12+(x(1)+x(3))^2/4));
alpha=6000/(sqrt(2)*x(1)*x(2));
beta=Q*D/J;
tau=sqrt(alpha^2+2*alpha*beta*x(2)/(2*D)+beta^2);
sigma=504000/(x(4)*x(3)^2);
delta=65856000/(30*10^6*x(4)*x(3)^3);
F=4.013*(30*10^6)/196*sqrt(x(3)^2*x(4)^6/36)*(1-x(3)*sqrt(30/48)/28);
g(1)=tau-13600;
g(2)=sigma-30000;
g(3)=x(1)-x(4);
g(4)=0.10471*x(1)^2+0.04811*x(3)*x(4)*(14+x(2))-5.0;
g(5)=0.125-x(1);
g(6)=delta-0.25;
g(7)=6000-F;
% Equal(ity constraints
geq=[];
return