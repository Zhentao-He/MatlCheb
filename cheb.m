function [D,x] = cheb(N,x1,x2)
% CHEB compute D = differentiation matrix, x = Chebyshev grid
if N==0,D=0;x=1;return,end

x = cos(pi*(0:N)/N)'; % [-1,1]
c = [2;ones(N-1,1);2].*(-1).^(0:N)';
X = repmat(x,1,N+1);
dX = X-X';
D = (c*(1./c)')./(dX+(eye(N+1)));
D = D-diag(sum(D,2));

if nargin == 3
    L = x2-x1;
    x = L/2*(x+1) + x1;
    D = D/(L/2);
end