# Pattern-Fill: Algorithm Description

## 0. Shared Preprocessing — Daily Profile Extraction

Before fitting any pattern, the raw time series $\{(t_i, y_i)\}$ is reduced to a **daily profile** $\bar{y}(h)$, where $h \in [0, 24)$ is the fractional hour-of-day.

The timeline is partitioned into bins of width $\Delta h = \Delta_{\min}/60$ hours (default $\Delta_{\min} = 15$ min), giving $B = 24/\Delta h$ bins. For each bin $b$, the representative value is:

$$\bar{y}_b = \underset{i:\, t_i \bmod 24 \in \text{bin } b}{\operatorname{median}}(y_i)$$

(or mean, depending on user choice). The result is a low-dimensional, noise-resistant summary of the diurnal shape, indexed by bin-center hours.

> **What data is used?** All available (non-NaN) observations across the entire series — no windowing is applied at this stage.

---

## 1. Spline Pattern Fill

### 1.1 Pattern Representation

A `DailyPattern` in spline mode is a **periodic natural cubic spline** $s(h)$ defined by $K$ user-supplied or automatically-selected control points $\{(h_k, v_k)\}_{k=1}^K$ with $h_k \in [0, 24)$.

To enforce 24-hour periodicity, the endpoint $(24, v_1)$ is appended before constructing the spline, and `scipy.interpolate.CubicSpline` is invoked with `bc_type="periodic"`. This imposes:

$$s'(0) = s'(24), \qquad s''(0) = s''(24)$$

On each interval $[h_k, h_{k+1}]$ the spline takes the form:

$$s(h) = a_k + b_k(h - h_k) + c_k(h - h_k)^2 + d_k(h - h_k)^3$$

with coefficients $a_k, b_k, c_k, d_k$ determined by the $C^2$ continuity and periodicity conditions. The output is clipped to $[0, 1]$.

### 1.2 Auto-Calibration (`fit_pattern`)

When the pattern is fitted from data:

1. Extract the daily profile $\bar{y}_b$ from **all clean data** in the series.
2. Select $K$ evenly-spaced control-point hours: $h_k = \frac{24(k-1)}{K}$, $k = 1,\ldots,K$ (default $K = 8$).
3. Interpolate the profile to obtain $v_k = \bar{y}(h_k)$.
4. Min–max normalize:

$$v_k^* = \frac{v_k - \min_k v_k}{\max_k v_k - \min_k v_k}$$

> **Calibration data:** The entire non-missing record is used (global fit, no windowing).

---

## 2. Sine Pattern Fill

### 2.1 Pattern Representation

The pattern is a sum of $M$ cosine components plus a DC offset (baseline):

$$p(h) = b + \sum_{m=1}^{M} A_m \cos\!\left(\frac{2\pi f_m (h - \phi_m)}{24}\right)$$

where $b \in [0,1]$ is the baseline, and for each component $m$: $A_m \geq 0$ is the amplitude, $f_m > 0$ is frequency in cycles per day, and $\phi_m \in [0, 24)$ is the phase in hours (the cosine formulation places the maximum of component $m$ at $h = \phi_m$). The output is clipped to $[0, 1]$.

### 2.2 Auto-Calibration (`fit_sine_pattern`)

**Step 1 — Daily profile.** Extract $\bar{y}(h)$ from all clean data (global). Min–max normalize to $\tilde{y}(h) \in [0,1]$.

**Step 2 — Frequency detection via FFT** (if frequencies are not fixed):

The mean-centered profile $\tilde{y}(h) - \overline{\tilde{y}}$ is transformed:

$$\hat{Y}_k = \sum_{b=0}^{B-1} \tilde{y}_b \, e^{-2\pi i k b / B}$$

Frequencies in cycles per day are $f_k = 24 k / (B \cdot \Delta h)$. The $n$ bins with the largest magnitudes $|\hat{Y}_k|$ (excluding the DC bin $k=0$) are selected, then rounded to the nearest half-integer cycle per day for interpretability.

**Step 3 — Amplitude and phase fitting.** For each detected frequency $f_m$, the following nonlinear least-squares problem is solved over the daily profile:

$$\min_{A_m,\, \phi_m} \sum_b \left[\tilde{y}(h_b) - \operatorname{clip}\!\left(b + A_m \cos\!\left(\frac{2\pi f_m(h_b - \phi_m)}{24}\right),\, 0, 1\right)\right]^2$$

with bounds $A_m \in [0, 1]$, $\phi_m \in [0, 24]$. The baseline $b$ is fixed to the mean of the (unwindowed) daily profile. Initial guesses are $A_m^{(0)} = \sigma(\tilde{y} - b)$ and $\phi_m^{(0)} = h_{\arg\max \tilde{y}}$.

> **Calibration data:** All clean observations are used. The profile is a global median/mean across the full record.

---

## 3. Two-Pass Gap-Fill Architecture

`pattern_fill()` operates in two separated passes per series.

### Pass 1 — Pattern Fill

For each interior gap `[start, stop)`:

1. **Backward-window scaling.** Clean (non-NaN) samples are collected from the
   *original* (unfilled) series in a backward window ending at the gap start,
   controlled by `pattern_window_days`:
   - `None` (default): all clean samples in the series (global).
   - `n` days: samples in `[gap_start − n·days, gap_start)`.
   If fewer than 2 samples are found, the window expands to all prior clean data.

2. **Pattern scaling.** The 0–1 normalised pattern is mapped to the window range:
   $$ p_{\text{scaled}}(t) = p(h_t) \cdot (y_{\max} - y_{\min}) + y_{\min} $$

3. **Optional area normalisation** (`normalize_area=True`): rescale fill so its
   integral matches the expected daily-profile integral (see §4 below).

4. **Cosine boundary blending** (see §4): ensures $C^0$ continuity with the
   surrounding data.

### Pass 2 — AR Noise Augmentation (optional, `add_noise=True`)

After all gaps are filled by Pass 1, a second loop over the same gaps adds
stochastic variability. For each gap:

1. **Backward-window residuals.** A separate `ar_window_days` parameter
   (same fallback logic, min count = `ar_order + 1`) selects clean samples from
   the *original* (unfilled) series.  The pattern prediction uses the *window's
   own* min/max for scaling — not the global range — eliminating the scaling
   mismatch present in previous versions:
   $$ \hat{y}(t) = p(h_t) \cdot (y_{\max}^{\text{win}} - y_{\min}^{\text{win}}) + y_{\min}^{\text{win}} $$
   $$ \epsilon_t = y_t^{\text{original}} - \hat{y}(t) $$

2. **AR model fitting** via Yule-Walker equations (see §4 below).

3. **Noise generation and sine taper** (see §4.2 and §4.3 below).

---

## 4. AR Noise Augmentation Details

### 4.1 Residual Computation and AR Model Fitting

An AR($p$) model is fitted to the window residuals $\{\epsilon_t\}$ via the
**Yule–Walker equations**. The biased sample autocorrelation at lag $k$ is:

$$\hat{r}(k) = \frac{1}{|\mathcal{W}|} \sum_{t=0}^{|\mathcal{W}|-k-1} \epsilon_t \epsilon_{t+k}$$

The Toeplitz system $\mathbf{R}\,\boldsymbol{\phi} = \mathbf{r}$ is then solved:

$$\begin{bmatrix} \hat{r}(0) & \hat{r}(1) & \cdots & \hat{r}(p-1) \\ \hat{r}(1) & \hat{r}(0) & \cdots & \hat{r}(p-2) \\ \vdots & & \ddots & \vdots \\ \hat{r}(p-1) & \cdots & \hat{r}(1) & \hat{r}(0) \end{bmatrix} \begin{bmatrix} \phi_1 \\ \phi_2 \\ \vdots \\ \phi_p \end{bmatrix} = \begin{bmatrix} \hat{r}(1) \\ \hat{r}(2) \\ \vdots \\ \hat{r}(p) \end{bmatrix}$$

yielding AR coefficients $\phi_1, \ldots, \phi_p$.

> **Calibration data:** `ar_window_days=None` (default) → all clean data before
> the gap; `ar_window_days=n` → up to *n* days of clean data before the gap,
> expanding if fewer than `ar_order + 1` samples are available.

### 4.2 Noise Generation

The local residual std is $\hat{\sigma}_{\text{local}} = \operatorname{std}(\{\epsilon_t\})$. Because the residuals are computed in the *local* scaling of the AR window while the fill values live in the *fill* scaling, the noise std is rescaled before synthesis:

$$\hat{\sigma} = \hat{\sigma}_{\text{local}} \cdot \frac{y_{\max}^{\text{fill}} - y_{\min}^{\text{fill}}}{y_{\max}^{\text{win}} - y_{\min}^{\text{win}}}$$

A noise realization of length $N_{\text{gap}}$ is then drawn:

$$x_t = \sum_{i=1}^{p} \phi_i\, x_{t-i} + \varepsilon_t, \qquad \varepsilon_t \overset{\mathrm{iid}}{\sim} \mathcal{N}(0,\, \hat{\sigma}^2)$$

with a $2p$-sample warm-up period discarded to ensure approximate stationarity.

### 4.3 Boundary Taper

To prevent discontinuities at gap edges, the noise is multiplied by a half-sine taper. Let $d_t = \min(t,\, N_{\text{gap}}-1-t)$ be the distance from the nearer boundary, and define:

$$\tau_t = \sin\!\left(\frac{\pi}{2} \cdot \operatorname{clip}\!\left(\frac{d_t}{n_{\text{blend}}}, 0, 1\right)\right)$$

The tapered noise added to the pattern is $x_t \cdot \tau_t$, which equals 0 at the edges and ramps up to full amplitude within $n_{\text{blend}}$ samples from each boundary.

---

## 5. Boundary Blending (Common to All Fills)

After the pattern (with or without noise) is evaluated, a **cosine-decay blend** ensures continuity with the surrounding data. Let $y_L$ and $y_R$ be weighted-average anchor values computed from up to $n_{\text{blend}}$ non-NaN samples on the left and right of the gap respectively (linearly weighted toward the gap edge).

The final fill is:

$$\tilde{f}_t = w_t \cdot \left[(1 - \alpha_t) y_L + \alpha_t y_R\right] + (1 - w_t) \cdot \hat{f}_t$$

where $\alpha_t = t / (N_{\text{gap}} - 1)$ is a linear interpolation weight, $\hat{f}_t$ is the scaled (and optionally noise-augmented) pattern value, and the edge weight decays as:

$$w_t = \frac{1}{2}\left(1 + \cos\!\left(\pi \cdot \operatorname{clip}\!\left(\frac{d_t}{n_{\text{blend}}}, 0, 1\right)\right)\right)$$

This is 1 at the gap boundary and smoothly approaches 0 at $n_{\text{blend}}$ samples inward, guaranteeing $C^0$ continuity.
