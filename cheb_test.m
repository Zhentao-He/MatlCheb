function [err,p] = cheb_test(f)
% 本函数利用FCT计算原数据f与Chebyshev插值多项式值p的误差err
% 并返回Chebyshev插值多项式在对应点处的值p
% 先不管N是奇数的情形
F = myFCT1(f); %对于当前版本，f非矩阵将直接报错退出

if isrow(f)
    N = length(f)-1;
else % f为列向量或矩阵
    [N,~]=size(f);
    N = N-1;
end

p = 2*myFCT1(F)/N;

err = abs(f-p);
tol = 1e-13;
if mean(err) > 1e-13
    warning("The error is not small enough! Tol = " + num2str(tol))
end