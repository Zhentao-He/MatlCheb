function p = cheb_to_real(a)
% 本函数根据展开系数计算Chebyshev插值多项式在Chebyshev grids上的值
% 使用FCT实现，O(N log N)
% a 为向量时, p为相同size的向量
% a 为矩阵时, 将每一列视为一组展开系数
flagrow = 0;
if isrow(a)
    a = a.'; % 系数转成列矢量统一处理
    flagrow = 1;
end

[N,~] = size(a);
N = N - 1;

if mod(N,2) == 0
    % Invert real_to_cheb: undo endpoint halving + scaling, then inverse DCT via FCT
    % real_to_cheb(f) = [FCT(f)*2/N with endpoints halved]
    % => inverse: undo halving*(N/2) → FCT → scale by 2/N
    F = a;
    F(1,:)     = N * a(1,:);          % undo /2 then *N/2 → *N
    F(end,:)   = N * a(end,:);        % same
    F(2:end-1,:) = (N/2) * a(2:end-1,:);  % undo *2/N → *N/2
    p = 2 * myFCT1(F) / N;
else
    p = cos( (pi*(0:N)'/ N) .* (0:N) ) * a;
end

if flagrow
    p = p.';
end

% old version (matrix multiply, O(N^2)):
% p = cos( (pi*(0:N)'/ N) .* (0:N) ) * a;

