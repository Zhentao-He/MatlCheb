function vals = cheb_interpolate(a,x1,x2,x)
% 本函数利用Chebyshev系数a计算在x处的插值
% a是系数构成的向量/matrix
% x为标量时,返回的vals为标量/列矢量
% x为向量时,vals为列矢量/矩阵,第i行对应x(i)的函数值/向量
% 带求的插值函数定义在[x1,x2]上
% 并且[x1,x2]到Chebyshev格点z\in[-1,1]的变换为伸缩平移 z(x) = 2*(x-x1)/xL-1

% updates:
% 01/08/2025: 支持a为矩阵,x为向量的情况, 并将for循环优化为矢量计算.
% 01/12: 当x为行向量时,输出的vals为行向量

if x1 >= x2  % 交换x1 x2,使得x1为较小值
    tmp = x1; x1 = x2; x2 = tmp;
    clear tmp
end
xL = x2 - x1;

if isrow(a) 
    a = a.'; % 系数转成列矢量统一处理
end
N = size(a,1) - 1;
n = (0:N).';
theta = acos(2*(x-x1)/xL-1); % x同size

flagrow = 0; % 非行向量
if isscalar(x)
    vals = a.*cos(n*theta);
    vals = sum(vals).'; % 返回的vals为标量/列矢量
    return
elseif iscolumn(x)
    theta = theta.'; % 转成行矢量统一处理
elseif isrow(x)
    flagrow = 1; 
end
ntheta = reshape(n .* theta,[N+1,1,length(x)]); %  列矢量n .* 行矢量theta = N*length(x)矩阵
vals = a.*cos(ntheta); %[N+1,a的行数,length(x)]维矩阵
vals = sum(vals); % 沿着第一维求和,[1,a的列数,length(x)]维矩阵
vals = reshape(vals,[size(a,2),length(x)]); % [a的列数,length(x)]维矩阵
if flagrow
    return
else
    vals = vals.';
end
% if iscolumn(a)
%     N = length(a) - 1;
%     for n = 0:N
%         vals = vals + a(n+1)*cos(n*theta);
%     end
% else
%     % when 'a' is a matrix
%     N = size(a,1) - 1;
%     n = (0:N).';
%     vals = a.*cos(n*theta);
%     vals = sum(vals);
% end