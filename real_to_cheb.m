function a = real_to_cheb(f)
% 本函数利用FCT计算Chebyshev多项式的展开系数a
% 输出结果a的size与f相同
% f为向量或矩阵
% 当f是矩阵时，将f的每一列视为待变换的列向量
% 列向量长度为N+1，其中N最好为2的幂次，但必须为偶数
% 先不管N为奇数的情形
% 系数顺序：模式数从前往后为0到N

% updates:
% 01/08/2025: f为行向量时,a也为行向量

if ~ismatrix(f)
    error("f is required to be a matrix in myFCT1(f)!")
end

if isrow(f)
    f = f'; flagrow = 1;
else
    flagrow = 0;
end

[N,~] = size(f); N = N-1;

if mod(N,2) == 0 % 如果N是偶数，则使用FCT计算
    a = myFCT1(f);
    a = 2*a/N;
    a(1,:) = a(1,:)/2;
    a(end,:) = a(end,:)/2;
else
    error("The length of f is required to be odd !")
    % n = 0:N;
    % [ii,nn] = meshgrid(n,n');
    % A = cos(ii.*nn*pi/N);
    % a = A\f;
end
if flagrow
    a = a.';
end


% 以下代码为陈旧版本
% function a=real_to_cheb(x,f)
% % 本函数利用DFT计算Chebyshev多项式的展开系数a
% % 函数值f为列向量
% % x为Chebyshev points，范围任意（不限于-1到1）
% a=0*f;
% N=length(f)-1;
% a(1)=sum(f) - 0.5*( f(1) + f(end) );
% a(1)=a(1)/N;
% a(end)= sum( (-1).^(0:N)' .*f ) - 0.5*( f(1) + f(end)*(-1)^N );% 默认f是列向量 
% a(end)=a(end)/N;
% theta=acos(x);
% for n=2:N
%     a(n)=f(1) + f(end)*(-1)^(n-1) + 2*sum( ...
%         f(2:end-1).* cos( (n-1)*theta(2:end-1) ) );
%     a(n)= a(n)/N;
% end

            

