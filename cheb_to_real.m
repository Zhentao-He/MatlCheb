function p = cheb_to_real(a)
% 本函数根据展开系数计算Chebyshev插值多项式在Chebyshev grids上的值
% a 为向量时, p为相同size的向量
% a 为矩阵时, 将每一列视为一组展开系数
flagrow = 0;
if isrow(a) 
    a = a.'; % 系数转成列矢量统一处理
    flagrow = 1;
end
 
N = size(a,1)-1;

p = cos( (pi*(0:N)'/ N) .* (0:N) ) * a;

if flagrow
    p = p.';
end

% old version
% N=length(a)-1;
% theta=pi*(0:N)'/N;%控制 返回结果p为列向量
% p=0*theta;
% for n=0:N
%     p= p + a(n+1)*cos(n*theta);
% end

