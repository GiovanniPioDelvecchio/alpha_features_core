#pragma once

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <functional>

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
using Vec   = std::vector<double>;
using Mat   = std::vector<Vec>;   // Mat[col][row] — column-major: Mat[ticker][date]
using VecI  = std::vector<int>;

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
static constexpr double kEps = 1e-8;

// ---------------------------------------------------------------------------
// Utility: safe divide / isnan
// ---------------------------------------------------------------------------
inline double safe_div(double a, double b) {
    if (std::abs(b) < kEps) return NaN;
    return a / b;
}

// ===========================================================================
//  1-D PRIMITIVES  (operate on a single ticker time-series)
// ===========================================================================

// --- delta ------------------------------------------------------------------
Vec ts_delta(const Vec& x, int period);

// --- delay (lag) ------------------------------------------------------------
Vec ts_delay(const Vec& x, int period);

// --- rolling sum ------------------------------------------------------------
Vec ts_sum(const Vec& x, int window);

// --- rolling mean -----------------------------------------------------------
Vec ts_mean(const Vec& x, int window);

// --- rolling std (ddof=1) ---------------------------------------------------
Vec ts_std(const Vec& x, int window);

// --- rolling max / min ------------------------------------------------------
Vec ts_max(const Vec& x, int window);
Vec ts_min(const Vec& x, int window);

// --- rolling product --------------------------------------------------------
Vec ts_prod(const Vec& x, int window);

// --- ts rank (rank of last element in window) -------------------------------
Vec ts_rank(const Vec& x, int window);

// --- rolling correlation (Pearson) ------------------------------------------
Vec ts_corr(const Vec& x, const Vec& y, int window);

// --- rolling covariance -----------------------------------------------------
Vec ts_cov(const Vec& x, const Vec& y, int window);

// --- EWM / SMA  (alpha = m/n, adjust=False) ---------------------------------
Vec ts_sma(const Vec& x, int n, int m);

// --- WMA (decay 0.9) --------------------------------------------------------
Vec ts_wma(const Vec& x, int window);

// --- decay linear (linearly weighted) ---------------------------------------
Vec ts_decaylinear(const Vec& x, int window);

// --- OLS beta (x = Sequence(window)) ----------------------------------------
Vec ts_regbeta(const Vec& y, int window);

// --- count of True values ---------------------------------------------------
Vec ts_count(const Vec& cond, int window);   // cond: 1.0=true, 0.0=false, NaN=ignore

// --- sumif ------------------------------------------------------------------
Vec ts_sumif(const Vec& x, int window, const Vec& cond);

// --- lowday / highday (days since lowest / highest) -------------------------
Vec ts_lowday(const Vec& x, int window);
Vec ts_highday(const Vec& x, int window);

// --- log, sign, abs (element-wise) ------------------------------------------
Vec ew_log(const Vec& x);
Vec ew_sign(const Vec& x);
Vec ew_abs(const Vec& x);

// --- element-wise arithmetic ------------------------------------------------
Vec ew_add(const Vec& a, const Vec& b);
Vec ew_sub(const Vec& a, const Vec& b);
Vec ew_mul(const Vec& a, const Vec& b);
Vec ew_div(const Vec& a, const Vec& b);
Vec ew_pow(const Vec& a, const Vec& b);         // a^b element-wise
Vec ew_pow_scalar(const Vec& a, double b);
Vec ew_scalar_add(const Vec& a, double s);
Vec ew_scalar_mul(const Vec& a, double s);
Vec ew_scalar_div(double s, const Vec& a);       // s / a[i]
Vec ew_max2(const Vec& a, const Vec& b);         // element-wise max(a,b)
Vec ew_min2(const Vec& a, const Vec& b);

// ===========================================================================
//  2-D (CROSS-SECTIONAL) PRIMITIVES  — operate across tickers for each date
//  Mat layout: matrix[ticker][date]
// ===========================================================================

// --- cross-sectional rank (percentile, ties=min) ----------------------------
Mat cs_rank(const Mat& m);

// ===========================================================================
//  Alpha191 class
//  Inputs are passed as Mat (column-major: [ticker][date]).
//  All methods return Mat of the same shape.
// ===========================================================================
class Alpha191 {
public:
    // --- constructor --------------------------------------------------------
    Alpha191(
        const Mat& open,
        const Mat& high,
        const Mat& low,
        const Mat& close,
        const Mat& volume,
        const Mat& amount,       // close * volume
        const Mat& returns       // pct_change of close
    );

    // --- compute a single alpha (1-based index) -----------------------------
    Mat calculate(int alpha_num) const;

    // --- compute all alphas; returns map alpha_name -> Mat ------------------
    std::unordered_map<std::string, Mat> calculate_all() const;

    // --- dimensions ---------------------------------------------------------
    int n_tickers() const { return static_cast<int>(close_.size()); }
    int n_dates()   const { return close_.empty() ? 0 : static_cast<int>(close_[0].size()); }

    // individual alphas
    Mat alpha001() const;
    Mat alpha002() const;
    Mat alpha003() const;
    Mat alpha004() const;
    Mat alpha005() const;
    Mat alpha006() const;
    Mat alpha007() const;
    Mat alpha008() const;
    Mat alpha009() const;
    Mat alpha010() const;
    Mat alpha011() const;
    Mat alpha012() const;
    Mat alpha013() const;
    Mat alpha014() const;
    Mat alpha015() const;
    Mat alpha016() const;
    Mat alpha017() const;
    Mat alpha018() const;
    Mat alpha019() const;
    Mat alpha020() const;
    Mat alpha021() const;
    Mat alpha022() const;
    Mat alpha023() const;
    Mat alpha024() const;
    Mat alpha025() const;
    Mat alpha026() const;
    Mat alpha027() const;
    Mat alpha028() const;
    Mat alpha029() const;
    Mat alpha030() const;
    Mat alpha031() const;
    Mat alpha032() const;
    Mat alpha033() const;
    Mat alpha034() const;
    Mat alpha035() const;
    Mat alpha036() const;
    Mat alpha037() const;
    Mat alpha038() const;
    Mat alpha039() const;
    Mat alpha040() const;
    Mat alpha041() const;
    Mat alpha042() const;
    Mat alpha043() const;
    Mat alpha044() const;
    Mat alpha045() const;
    Mat alpha046() const;
    Mat alpha047() const;
    Mat alpha048() const;
    Mat alpha049() const;
    Mat alpha050() const;
    Mat alpha051() const;
    Mat alpha052() const;
    Mat alpha053() const;
    Mat alpha054() const;
    Mat alpha055() const;
    Mat alpha056() const;
    Mat alpha057() const;
    Mat alpha058() const;
    Mat alpha059() const;
    Mat alpha060() const;
    Mat alpha061() const;
    Mat alpha062() const;
    Mat alpha063() const;
    Mat alpha064() const;
    Mat alpha065() const;
    Mat alpha066() const;
    Mat alpha067() const;
    Mat alpha068() const;
    Mat alpha069() const;
    Mat alpha070() const;
    Mat alpha071() const;
    Mat alpha072() const;
    Mat alpha073() const;
    Mat alpha074() const;
    Mat alpha075() const;
    Mat alpha076() const;
    Mat alpha077() const;
    Mat alpha078() const;
    Mat alpha079() const;
    Mat alpha080() const;
    Mat alpha081() const;
    Mat alpha082() const;
    Mat alpha083() const;
    Mat alpha084() const;
    Mat alpha085() const;
    Mat alpha086() const;
    Mat alpha087() const;
    Mat alpha088() const;
    Mat alpha089() const;
    Mat alpha090() const;
    Mat alpha091() const;
    Mat alpha092() const;
    Mat alpha093() const;
    Mat alpha094() const;
    Mat alpha095() const;
    Mat alpha096() const;
    Mat alpha097() const;
    Mat alpha098() const;
    Mat alpha099() const;
    Mat alpha100() const;
    Mat alpha101() const;
    Mat alpha102() const;
    Mat alpha103() const;
    Mat alpha104() const;
    Mat alpha105() const;
    Mat alpha106() const;
    Mat alpha107() const;
    Mat alpha108() const;
    Mat alpha109() const;
    Mat alpha110() const;
    Mat alpha111() const;
    Mat alpha112() const;
    Mat alpha113() const;
    Mat alpha114() const;
    Mat alpha115() const;
    Mat alpha116() const;
    Mat alpha117() const;
    Mat alpha118() const;
    Mat alpha119() const;
    Mat alpha120() const;
    Mat alpha121() const;
    Mat alpha122() const;
    Mat alpha123() const;
    Mat alpha124() const;
    Mat alpha125() const;
    Mat alpha126() const;
    Mat alpha127() const;
    Mat alpha128() const;
    Mat alpha129() const;
    Mat alpha130() const;
    Mat alpha131() const;
    Mat alpha132() const;
    Mat alpha133() const;
    Mat alpha134() const;
    Mat alpha135() const;
    Mat alpha136() const;
    Mat alpha137() const;
    Mat alpha138() const;
    Mat alpha139() const;
    Mat alpha140() const;
    Mat alpha141() const;
    Mat alpha142() const;
    Mat alpha143() const;
    Mat alpha144() const;
    Mat alpha145() const;
    Mat alpha146() const;
    Mat alpha147() const;
    Mat alpha148() const;
    Mat alpha149() const;
    Mat alpha150() const;
    Mat alpha151() const;
    Mat alpha152() const;
    Mat alpha153() const;
    Mat alpha154() const;
    Mat alpha155() const;
    Mat alpha156() const;
    Mat alpha157() const;
    Mat alpha158() const;
    Mat alpha159() const;
    Mat alpha160() const;
    Mat alpha161() const;
    Mat alpha162() const;
    Mat alpha163() const;
    Mat alpha164() const;
    Mat alpha165() const;
    Mat alpha166() const;
    Mat alpha167() const;
    Mat alpha168() const;
    Mat alpha169() const;
    Mat alpha170() const;
    Mat alpha171() const;
    Mat alpha172() const;
    Mat alpha173() const;
    Mat alpha174() const;
    Mat alpha175() const;
    Mat alpha176() const;
    Mat alpha177() const;
    Mat alpha178() const;
    Mat alpha179() const;
    Mat alpha180() const;
    Mat alpha181() const;
    Mat alpha182() const;
    Mat alpha183() const;
    Mat alpha184() const;
    Mat alpha185() const;
    Mat alpha186() const;
    Mat alpha187() const;
    Mat alpha188() const;
    Mat alpha189() const;
    Mat alpha190() const;
    Mat alpha191() const;

private:
    Mat open_, high_, low_, close_, volume_, amount_, returns_;
    Mat vwap_;          // amount / volume

    // --- per-ticker helpers -------------------------------------------------
    Mat mat_apply1(const Mat& a,
                   std::function<Vec(const Vec&)> fn) const;
    Mat mat_apply2(const Mat& a, const Mat& b,
                   std::function<Vec(const Vec&, const Vec&)> fn) const;

    // --- helpers that mirror pandas primitives ------------------------------
    Mat Delay(const Mat& x, int period) const;
    Mat Delta(const Mat& x, int period) const;
    Mat Sum(const Mat& x, int window) const;
    Mat Mean(const Mat& x, int window) const;
    Mat Std(const Mat& x, int window) const;
    Mat Tsmax(const Mat& x, int window) const;
    Mat Tsmin(const Mat& x, int window) const;
    Mat Tsrank(const Mat& x, int window) const;
    Mat Corr(const Mat& x, const Mat& y, int window) const;
    Mat Cov(const Mat& x, const Mat& y, int window) const;
    Mat Sma(const Mat& x, int n, int m) const;
    Mat Wma(const Mat& x, int window) const;
    Mat Decaylinear(const Mat& x, int window) const;
    Mat Regbeta(const Mat& y, int window) const;
    Mat Count(const Mat& cond, int window) const;
    Mat Sumif(const Mat& x, int window, const Mat& cond) const;
    Mat Lowday(const Mat& x, int window) const;
    Mat Highday(const Mat& x, int window) const;
    Mat Log(const Mat& x) const;
    Mat Sign(const Mat& x) const;
    Mat Abs(const Mat& x) const;
    // element-wise binary
    Mat Add(const Mat& a, const Mat& b) const;
    Mat Sub(const Mat& a, const Mat& b) const;
    Mat Mul(const Mat& a, const Mat& b) const;
    Mat Div(const Mat& a, const Mat& b) const;
    Mat Pow(const Mat& a, const Mat& b) const;
    Mat ScalarAdd(const Mat& a, double s) const;
    Mat ScalarMul(const Mat& a, double s) const;
    Mat ScalarDiv(double s, const Mat& a) const;   // s / a
    Mat EwMax(const Mat& a, const Mat& b) const;
    Mat EwMin(const Mat& a, const Mat& b) const;
    Mat ScalarMax(const Mat& a, double s) const;   // max(a[i], s)
    Mat ScalarMin(const Mat& a, double s) const;
    // cross-sectional
    Mat Rank(const Mat& x) const;

    // conditional assignment helper
    // result[t][d] = a[t][d] if cond[t][d]>0.5 else b[t][d]
    Mat Where(const Mat& cond, const Mat& a, const Mat& b) const;
    // Where with scalar branches
    Mat WhereScalar(const Mat& cond, double a, double b) const;
    Mat WhereLeft(const Mat& cond, const Mat& a, double b) const;
    Mat WhereRight(const Mat& cond, double a, const Mat& b) const;

    // zero-filled matrix
    Mat zeros() const;
    Mat scalar_mat(double v) const;
};