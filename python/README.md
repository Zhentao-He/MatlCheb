# Chebyshev pseudospectral routines (Python)

Python port of the MATLAB files in the repository root. Those files implement the Chebyshev pseudospectral method from Trefethen, *Spectral Methods in MATLAB*: a differentiation matrix on Chebyshev–Gauss–Lobatto nodes, a fast cosine transform, the map between grid samples and Chebyshev coefficients, evaluation of a Chebyshev series, analytic mesh refinement, and a coefficient-decay plot.

The original `.m` files are unchanged. Background and notation are in the repository `Readme.md`.

## Mapping / 文件对照

| MATLAB | Python module | Function |
| --- | --- | --- |
| `cheb.m` | `chebyshev/cheb.py` | `cheb` |
| `cheb_mp.m` | `chebyshev/cheb_mp.py` | `cheb_mp` |
| `myFCT1.m` | `chebyshev/fct.py` | `myFCT1` |
| `real_to_cheb.m` | `chebyshev/transform.py` | `real_to_cheb` |
| `cheb_to_real.m` | `chebyshev/transform.py` | `cheb_to_real` |
| `cheb_interpolate.m` | `chebyshev/interpolate.py` | `cheb_interpolate` |
| `cheb_test.m` | `chebyshev/cheb_test.py` | `cheb_test` |
| `AnMR.m` | `chebyshev/mesh.py` | `AnMR` |
| `spec_plot.m` | `chebyshev/plotting.py` | `spec_plot` |

## Install / 安装

From this directory:

```bash
pip install -r requirements.txt
```

Needs NumPy, SciPy, Matplotlib, and mpmath. Tests also need pytest. GNU Octave is used only for the comparison against the original `.m` files; the core tests run without it.

在本目录执行上面的安装命令。运行库需要 NumPy、SciPy、Matplotlib 和 mpmath；测试还需要 pytest。与原始 `.m` 文件对拍时才需要 GNU Octave，核心测试可以不装 Octave。

## Run each routine / 如何运行

Work from this directory so the package imports:

```bash
cd python
python -m chebyshev          # short demo of every routine / 每个函数的简短示例
python -m pytest             # analytic tests and, if installed, Octave comparison
```

If the interpreter is only available as `python3`, use that name instead. / 若环境里只有 `python3`，把上面的 `python` 换成 `python3`。

```python
import numpy as np
from chebyshev import (
    AnMR, cheb, cheb_interpolate, cheb_mp, cheb_test,
    cheb_to_real, myFCT1, real_to_cheb, spec_plot,
)

D, x = cheb(16)                  # cheb.m on [-1, 1]
D, x = cheb(16, 0.0, 1.0)        # same nodes, affine-mapped to [0, 1]
D_mp, x_mp = cheb_mp(16, dps=50) # cheb_mp.m via mpmath (not Advanpix mp)

f = np.exp(x)                    # samples at the Lobatto nodes, mode 0..N
F = myFCT1(f)                    # myFCT1.m; N must be even and positive
a = real_to_cheb(f)              # real_to_cheb.m, coefficients of T_n
f2 = cheb_to_real(a)             # cheb_to_real.m, back to the nodes
err, p = cheb_test(f)            # cheb_test.m, |f - 2*FCT(FCT(f))/N|

y = cheb_interpolate(a, -1.0, 1.0, np.linspace(-1.0, 1.0, 9))
Dxx, Dx, x_ref = AnMR(16, xB=1.0, kappa=0.5)  # AnMR.m
spec_plot(f)                     # spec_plot.m: markers of log10|a_n|
```

A 1-D NumPy array is treated as a MATLAB column. Shape `(n, 1)`, `(1, n)`, and `(n, m)` follow the MATLAB row / column / matrix rules, including the layout returned by `cheb_interpolate`.

一维 NumPy 数组按 MATLAB 列向量处理。形状 `(n, 1)`、`(1, n)`、`(n, m)` 与 MATLAB 的列、行、矩阵一致，`cheb_interpolate` 的输出方向也保持一致。

`kappa` in `AnMR` must be nonzero (`kappa = 0` is the plain `cheb` grid, but the formula divides by `sinh(2 kappa)`). `xB` is `+1` or `-1`, the endpoint where nodes cluster.

## What the port checks / 验证

`python -m pytest` checks:

- `cheb`: row sums vanish; derivatives of polynomials of degree `<= N` are exact; off-diagonal entries match Trefethen’s closed form.
- `myFCT1`, `real_to_cheb`, `cheb_to_real`: round trip; coefficients of `T_n` are unit vectors; the cosine sum and `scipy.fft.dct(..., type=1)` agree.
- `cheb_interpolate`: `T_2` and a mapped linear series, including endpoint clipping.
- `AnMR`: chain rule on polynomials of the reference coordinate `y`, which the matrix differentiates exactly.
- `cheb_mp`: agreement with float64 `cheb` and with the closed form at 40 decimal places (absolute error below `1e-30`).
- `spec_plot`: the marker y-data are `log10|a_n|`.

Against GNU Octave 8.4, driving the original `.m` files on the same inputs (`tests/test_octave.py`, `tests/octave_reference.m`):

| Comparison | Tolerance |
| --- | --- |
| `cheb` matrix and nodes, `N <= 32`, including an affine map | `rtol=1e-11`, `atol=1e-9` |
| `cheb`, `N = 64` | `rtol=1e-10`, `atol=1e-8` |
| `myFCT1`, `real_to_cheb`, `cheb_to_real`, `cheb_test`, `cheb_interpolate` | `rtol=1e-11`, `atol=1e-11` (`5e-10` absolute at `N = 64`) |
| `AnMR` nodes | `rtol=1e-11`, `atol=1e-11` |
| `AnMR` first-derivative matrix | `rtol=1e-10`, `atol=1e-9` |
| `AnMR` second-derivative matrix | `rtol=1e-10`, `atol=1e-8` |
| series drawn by `spec_plot` | `rtol=1e-11`, `atol=1e-11` |

On that Octave suite the largest absolute difference was `1.8e-12` (`AnMR` second-derivative matrix, `N = 16`, `kappa = 0.8`). Grids matched exactly or to a few units in the last place. Transforms, interpolation, and the `spec_plot` series stayed at or below about `1e-15`.

`cheb_mp.m` calls `mp` (Advanpix-style multiprecision). Octave does not provide `mp`, so that file is not executed there. `cheb_mp` is checked with mpmath against `cheb` (max absolute difference `7e-15` at `N = 8`, 40 decimal places) and against the closed-form differentiation matrix (absolute error below `1e-30`).

在上述 Octave 对拍中，绝对误差最大为 `1.8e-12`（`AnMR` 的二阶导数矩阵，`N = 16`，`kappa = 0.8`）。网格点完全一致或只差几个末位。变换、插值和 `spec_plot` 所用序列的误差大约在 `1e-15` 或更小。

`cheb_mp.m` 调用的是多精度函数 `mp`（Advanpix 风格）。Octave 没有 `mp`，因此没有直接运行该文件。`cheb_mp` 用 mpmath 与 `cheb` 对比（`N = 8`、40 位小数时最大绝对误差 `7e-15`），并与微分矩阵的闭合公式对比（绝对误差小于 `1e-30`）。

## Intentional differences / 与 MATLAB 的少数差别

- Passing only one endpoint to `cheb` raises `ValueError`. MATLAB ignored a lone extra argument and stayed on `[-1, 1]`.
- `cheb` and `cheb_interpolate` reject `x1 == x2` instead of dividing by zero.
- `AnMR(..., kappa=0)` raises. The MATLAB expression divides by `sinh(0)`; use `cheb` for that grid.
- `myFCT1` rejects `N <= 0` with an error. MATLAB hits an out-of-bounds index. Odd `N` still raises `N has to be even.`
- `cheb_mp` is mpmath, not the Advanpix `mp` toolbox. Results are `mpmath` matrices. Do further arithmetic inside `mp.workdps(dps)`.
- `spec_plot` adds axis labels and a title. The plotted numbers are the same `log10|real_to_cheb(f)|` series. `show=False` by default so batch runs do not block.
- `cheb` and `AnMR` return the node vector as a 1-D array. The entries match MATLAB’s column.

`cheb_to_real` still uses the cosine matrix when `N` is odd, as in the MATLAB file. `real_to_cheb` still requires an odd sample length (`N` even).

上述差别的中文说明：

- `cheb` 只给一个端点时会报错；MATLAB 会忽略这个多余参数，仍在 `[-1, 1]` 上计算。
- `x1 == x2` 时 `cheb` 和 `cheb_interpolate` 直接报错，而不是除以零。
- `AnMR` 在 `kappa = 0` 时报错。MATLAB 公式会除以 `sinh(0)`；不加密网格时请用 `cheb`。
- `myFCT1` 在 `N <= 0` 时给出明确错误；MATLAB 会越界。`N` 为奇数时仍报 `N has to be even.`
- `cheb_mp` 使用 mpmath，不是 Advanpix 的 `mp`。返回值是 mpmath 矩阵，后续运算请放在 `mp.workdps(dps)` 里。
- `spec_plot` 增加了坐标轴标签和标题，绘制的数值仍是 `log10|real_to_cheb(f)|`。默认 `show=False`，批处理时不会卡住。
- `cheb` 和 `AnMR` 的节点用一维数组返回，数值与 MATLAB 的列向量相同。
