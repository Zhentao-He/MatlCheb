% Shared inputs from the Python tests; writes MATLAB/Octave results.
% Usage: octave --no-gui --quiet octave_reference.m input.mat output.mat repo_root
args = argv();
in_path = args{1};
out_path = args{2};
repo_root = args{3};
addpath(repo_root);

S = load(in_path);
f16 = S.f16;
f16_row = S.f16_row;
f16_mat = S.f16_mat;
f8 = S.f8;
f64 = S.f64;
a_odd = S.a_odd;
ac = S.ac;
ar = S.ar;
Amat = S.Amat;
xs = S.xs;
xcol = S.xcol;
xrow = S.xrow;
x_domain = S.x_domain;

[D16, x16] = cheb(16);
[D32, x32] = cheb(32);
[D64, x64] = cheb(64);
[D1, x1] = cheb(1);
[D0, x0] = cheb(0);
[Dmap, xmap] = cheb(16, -2, 5);
[Drev, xrev] = cheb(8, 3, -1);

F16 = myFCT1(f16);
F16_row = myFCT1(f16_row);
F16_mat = myFCT1(f16_mat);
F8 = myFCT1(f8);
F64 = myFCT1(f64);

a16 = real_to_cheb(f16);
a16_row = real_to_cheb(f16_row);
a16_mat = real_to_cheb(f16_mat);
a8 = real_to_cheb(f8);
a64 = real_to_cheb(f64);

p16 = cheb_to_real(a16);
p16_row = cheb_to_real(a16_row);
p16_mat = cheb_to_real(a16_mat);
p_from_user = cheb_to_real(ac);
p_odd = cheb_to_real(a_odd);

[err16, ptest16] = cheb_test(f16);
[err_row, ptest_row] = cheb_test(f16_row);
[err_mat, ptest_mat] = cheb_test(f16_mat);

v_s = cheb_interpolate(ac, -1, 1, xs);
v_col = cheb_interpolate(ac, -1, 1, xcol);
v_row = cheb_interpolate(ac, -1, 1, xrow);
v_domain = cheb_interpolate(ac, 0, 4, x_domain);
v_swap = cheb_interpolate(ac, 4, 0, x_domain);
v_mat_s = cheb_interpolate(Amat, -1, 1, xs);
v_mat_col = cheb_interpolate(Amat, -1, 1, xcol);
v_mat_row = cheb_interpolate(Amat, -1, 1, xrow);
v_row_a = cheb_interpolate(ar, -1, 1, xcol);
v_clip = cheb_interpolate(ac, -1, 1, 1.0000001);

[Dxx, Dx, xam] = AnMR(16, 1, 0.8);
[Dxx2, Dx2, xam2] = AnMR(12, -1, 1.5);
[Dxx4, Dx4, xam4] = AnMR(4, 1, 0.7);

spec = log10(abs(real_to_cheb(f16)));

spec_plot_ok = 1;
spec_plot_msg = "";
try
    set(0, "defaultfigurevisible", "off");
    spec_plot(f16);
    close all;
catch err
    spec_plot_ok = 0;
    spec_plot_msg = err.message;
end

save("-v7", out_path, ...
    "D16", "x16", "D32", "x32", "D64", "x64", "D1", "x1", "D0", "x0", ...
    "Dmap", "xmap", "Drev", "xrev", ...
    "F16", "F16_row", "F16_mat", "F8", "F64", ...
    "a16", "a16_row", "a16_mat", "a8", "a64", ...
    "p16", "p16_row", "p16_mat", "p_from_user", "p_odd", ...
    "err16", "ptest16", "err_row", "ptest_row", "err_mat", "ptest_mat", ...
    "v_s", "v_col", "v_row", "v_domain", "v_swap", ...
    "v_mat_s", "v_mat_col", "v_mat_row", "v_row_a", "v_clip", ...
    "Dxx", "Dx", "xam", "Dxx2", "Dx2", "xam2", "Dxx4", "Dx4", "xam4", ...
    "spec", "spec_plot_ok", "spec_plot_msg");
