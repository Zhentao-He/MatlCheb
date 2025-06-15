% function [Dx,x] = AnMR(N,xB,kappa)
% % xB = -1 或 1, 为需要更密集的那一端
% % kappa > 0, 越大越密集, kappa=0 退回为cheb
% 
% [Dy,y] = cheb(N);
% x = xB*(1 - 2*sinh(kappa*(1-xB*y))/sinh(2*kappa));
% xy = kappa * xB^2 * 2 * cosh(kappa*(1-xB*y)) / sinh(2*kappa); % col vec
% Dx = Dy./xy;
% end
function [Dxx,Dx,x] = AnMR(N,xB,kappa)
% This function performs Analytic mesh-refinement
% xB = \pm1, is near the steep region.
% steeper, larger kappa

[Dy,y] = cheb(N);
xy = kappa * xB^2 * 2 * cosh(kappa*(1-xB*y)) / sinh(2*kappa); % col vec
xyy = -2*kappa^2 * xB^3 * csch(2*kappa) * sinh(kappa*(1-xB*y));

x = xB*(1 - 2*sinh(kappa*(1-xB*y))/sinh(2*kappa));
Dx = Dy./xy;
Dxx = - Dy .* xyy ./ xy.^3 + Dy^2 ./ xy.^2;