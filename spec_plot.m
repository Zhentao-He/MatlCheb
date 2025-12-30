function [] = spec_plot(f)
    % figure
    plot(log10(abs(real_to_cheb(f))),'.',MarkerSize=10);
end