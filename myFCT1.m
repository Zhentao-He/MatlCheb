function F = myFCT1(f)
% 利用FFT计算real data array f的第一类余弦变换.
% 输出结果F的size与f相同:
% 当f是矩阵时, 并返回每列的FCT
% 系数顺序：模式数从前往后为0到N

% updates: 
% 01/08/2025: 更新了注释

%% 本程序暂不支持对高维数组进行FCT
if ~ismatrix(f)
    error("f is required to be a matrix in myFCT1(f)!")
end

%% 判断f是向量还是矩阵
typeflag = 1; %代表f为向量
N = length(f) - 1; %对于多维数组f, length(f)返回max(size(f))
F = 0*f;

if iscolumn(f) % 如果f是列向量
    jj = (0:N-1)';
elseif isrow(f)
    jj = (0:N-1);
else
    typeflag = 0; %代表f非向量
end

%% 分f向量/矩阵两种情况处理
if typeflag==1 % f是向量
    
    % 后面使用Matlab自带的fft函数, 故N不是非得为2的幂次
    % if mod(log2(N),1)~=0
    %     error("N is not a power of 2, so the FFT cannot be performed!")
    % end
    if mod(N,2)~=0
        error("N has to be even.")
    end
    
    %构造辅助变量
    y = 0.5*( f(1:end-1) + flip( f(2:end) ) )...
        -sin(jj*pi/N) .* ( f(1:end-1) - flip(f(2:end)) ); %第二项
    u = fft(y); % 排列顺序为k从0到N-1
    R = real(u);
    I = -imag(u);%Matlab中FFT的定义为exp(-2*pi*i jk/N)
    % 以下代码默认N为偶数
    kk = 0:N/2;
    F(2*kk+1) = R(kk+1);%+1是因为Matlab指标从1开始
    j = jj(2:end); % j从1取到N-1
    F(1+1) = 0.5*(f(1)-f(end)) + sum( f(j+1).*cos(j*pi/N) );
    for k = 1:(N/2-1)
        F(2*k+1+1)=F(2*k)+I(k+1);
    end

else % f是矩阵

    [N,~] = size(f); %将f的各列视为向量，并返回每列的FCT
    N = N-1;
    % if mod(log2(N),1)~=0
    %     error("N is not a power of 2, so the FFT cannot be performed!")
    % end
    if mod(N,2)~=0
        error("N has to be even.")
    end
    jj=(0:N-1)';
    %构造辅助变量
    y = 0.5*( f(1:end-1,:) + flip( f(2:end,:) ) )...
        -sin(jj*pi/N) .* ( f(1:end-1,:) - flip(f(2:end,:)) );
    u = fft(y); % 排列顺序为k从0到N-1
    R = real(u);
    I = -imag(u);%Matlab中FFT的定义为exp(-2*pi*i jk/N)
    % 以下代码默认N为偶数
    kk = 0:N/2;
    F(2*kk+1,:) = R(kk+1,:);%+1是因为Matlab指标从1开始
    j = jj(2:end); % j从1取到N-1
    F(1+1,:) = 0.5*(f(1,:)-f(end,:)) + sum( f(j+1,:).*cos(j*pi/N) );
    for k = 1:(N/2-1)
        F(2*k+1+1,:) = F(2*k,:)+I(k+1,:);
    end
end
