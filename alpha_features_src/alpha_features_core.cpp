#include "alpha_features_core.h"
#include <cassert>
#include <cstring>
#include <stdexcept>

// ===========================================================================
//  1-D PRIMITIVES
// ===========================================================================

Vec ts_delta(const Vec& x, int period) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = period; i < n; ++i)
        out[i] = x[i] - x[i - period];
    return out;
}

Vec ts_delay(const Vec& x, int period) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = period; i < n; ++i)
        out[i] = x[i - period];
    return out;
}

Vec ts_sum(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    double s = 0.0;
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(x[i])) { s += x[i]; ++cnt; }
        if (i >= window) {
            if (!std::isnan(x[i - window])) { s -= x[i - window]; --cnt; }
        }
        if (i >= window - 1 && cnt == window) out[i] = s;
        else if (i >= window - 1) out[i] = NaN;
    }
    return out;
}

Vec ts_mean(const Vec& x, int window) {
    auto s = ts_sum(x, window);
    for (auto& v : s) if (!std::isnan(v)) v /= window;
    return s;
}


Vec ts_std(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        double sum = 0, sum2 = 0;
        int cnt = 0;
        for (int j = i - window + 1; j <= i; ++j) {
            if (!std::isnan(x[j])) { sum += x[j]; sum2 += x[j]*x[j]; ++cnt; }
        }
        if (cnt < 2) continue;
        //if (cnt < window) continue;
        double mean = sum / cnt;
        double var = (sum2 / cnt) - mean*mean;
        // unbiased (ddof=1)
        var = var * cnt / (cnt - 1);
        out[i] = var > 0 ? std::sqrt(var) : 0.0;
    }
    return out;
}


Vec ts_max(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        double mx = -std::numeric_limits<double>::infinity();
        int cnt = 0;  // FIX: conta solo validi
        for (int j = i - window + 1; j <= i; ++j) {
            if (!std::isnan(x[j])) { mx = std::max(mx, x[j]); cnt++; }
        }
        if (cnt == window) out[i] = mx;  // FIX: richiede finestra completa
    }
    return out;
}


Vec ts_min(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        double mn = std::numeric_limits<double>::infinity();
        int cnt = 0;
        for (int j = i - window + 1; j <= i; ++j) {
            if (!std::isnan(x[j])) { mn = std::min(mn, x[j]); cnt++; }
        }
        if (cnt == window) out[i] = mn;  // richiede finestra completa
    }
    return out;
}


Vec ts_prod(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        double p = 1.0;
        bool any = false;
        for (int j = i - window + 1; j <= i; ++j) {
            if (!std::isnan(x[j])) { p *= x[j]; any = true; }
        }
        if (any) out[i] = p;
        //if (cnt == window) out[i] = p;
    }
    return out;
}



Vec ts_rank(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        double last = x[i];
        if (std::isnan(last)) continue;
        // Verifica che tutti i valori nella finestra siano validi
        bool all_valid = true;
        for (int j = i - window + 1; j < i; ++j) {
            if (std::isnan(x[j])) { all_valid = false; break; }
        }
        if (!all_valid) continue;  // <-- aggiunto
        double rank = 1.0;
        for (int j = i - window + 1; j < i; ++j) {
            if (x[j] < last) rank += 1.0;
        }
        out[i] = rank;
    }
    return out;
}


/*
Vec ts_rank(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);

    for (int i = window - 1; i < n; ++i) {

        double last = x[i];
        if (std::isnan(last)) continue;

        int less = 0;
        int equal = 0;

        for (int j = i - window + 1; j <= i; ++j) {
            if (std::isnan(x[j])) { less = -1; break; }
            if (x[j] < last) less++;
            else if (x[j] == last) equal++;
        }

        if (less < 0) continue;

        double rank = less + 0.5 * equal;

        out[i] = rank / window;
    }

    return out;
}
*/


Vec ts_corr(const Vec& x, const Vec& y, int window) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        double sx=0, sy=0, sxy=0, sx2=0, sy2=0;
        int cnt = 0;
        for (int j = i - window + 1; j <= i; ++j) {
            if (std::isnan(x[j]) || std::isnan(y[j])) continue;
            sx += x[j]; sy += y[j];
            sxy += x[j]*y[j]; sx2 += x[j]*x[j]; sy2 += y[j]*y[j];
            ++cnt;
        }
        if (cnt < 2) { out[i] = 0.0; continue; }
        //if (cnt < window) continue;
        double num = cnt*sxy - sx*sy;
        double vx  = cnt*sx2 - sx*sx;
        double vy  = cnt*sy2 - sy*sy;
        // Distingui i casi come fa pandas:
        // vx==0 && vy==0 → entrambe costanti → pandas dà NaN (poi fillna→0)
        // vx==0 || vy==0 → una costante → pandas dà NaN (poi fillna→0)  
        // ma se il risultato è -inf → pandas lo mantiene come -inf!
        // In realtà pandas rolling corr dà -inf quando num!=0 ma den==0
        if (std::abs(vx) < kEps || std::abs(vy) < kEps) {
            if (std::abs(num) > 1e-12) {  // soglia molto bassa
                out[i] = (num > 0) ? std::numeric_limits<double>::infinity()
                                : -std::numeric_limits<double>::infinity();
            } else {
                out[i] = 0.0;
            }
            continue;
        }
        double r = num / std::sqrt(vx * vy);
        out[i] = std::max(-1.0, std::min(1.0, r));
    }
    return out;
}

Vec ts_cov(const Vec& x, const Vec& y, int window) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        double sx=0, sy=0, sxy=0;
        int cnt = 0;
        for (int j = i - window + 1; j <= i; ++j) {
            if (std::isnan(x[j]) || std::isnan(y[j])) continue;
            sx += x[j]; sy += y[j]; sxy += x[j]*y[j]; ++cnt;
        }
        if (cnt < 2) continue;
        out[i] = (sxy - sx*sy/cnt) / (cnt - 1);
    }
    return out;
}

// SMA: EWM with alpha=m/n, adjust=False  →  y[t] = alpha*x[t] + (1-alpha)*y[t-1]
Vec ts_sma(const Vec& x, int n, int m) {
    double alpha = static_cast<double>(m) / n;
    int sz = x.size();
    Vec out(sz, NaN);
    double val = NaN;
    for (int i = 0; i < sz; ++i) {
        if (std::isnan(x[i])) { out[i] = val; continue; }
        if (std::isnan(val)) val = x[i];
        else val = alpha * x[i] + (1.0 - alpha) * val;
        out[i] = val;
    }
    return out;
}

// WMA: weights = 0.9^(window-1-j), j=0..window-1
Vec ts_wma(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    Vec w(window);
    double sw = 0;
    for (int j = 0; j < window; ++j) {
        w[j] = std::pow(0.9, window - 1 - j);
        sw += w[j];
    }
    for (int i = window - 1; i < n; ++i) {
        double val = 0;
        for (int j = 0; j < window; ++j)
            val += w[j] * x[i - window + 1 + j];
        out[i] = val / sw;
    }
    return out;
}

// Decay linear: weights = 1,2,...,window (ascending), normalised
Vec ts_decaylinear(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    double sw = window * (window + 1) / 2.0;
    for (int i = window - 1; i < n; ++i) {
        double val = 0;
        bool any = true;
        for (int j = 0; j < window; ++j) {
            double v = x[i - window + 1 + j];
            if (std::isnan(v)) { any = false; break; }
            val += (j + 1) * v;
        }
        if (any) out[i] = val / sw;
    }
    return out;
}

// OLS beta: x_seq = 1..window (closed form)
Vec ts_regbeta(const Vec& y, int window) {
    int n = y.size();
    Vec out(n, NaN);
    // precompute constants for x = 1..window
    double sx  = window * (window + 1) / 2.0;
    double sx2 = window * (window + 1) * (2*window + 1) / 6.0;
    double denom = window * sx2 - sx * sx;
    if (std::abs(denom) < kEps) return out;
    for (int i = window - 1; i < n; ++i) {
        double sy = 0, sxy = 0;
        bool ok = true;
        for (int j = 0; j < window; ++j) {
            double v = y[i - window + 1 + j];
            if (std::isnan(v)) { ok = false; break; }
            sy  += v;
            sxy += (j + 1) * v;
        }
        if (ok) out[i] = (window * sxy - sx * sy) / denom;
    }
    return out;
}

Vec ts_count(const Vec& cond, int window) {
    int n = cond.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        double s = 0;
        for (int j = i - window + 1; j <= i; ++j) {
            if (!std::isnan(cond[j]) && cond[j] > 0.5) s += 1.0;
        }
        out[i] = s;
    }
    return out;
}

Vec ts_sumif(const Vec& x, int window, const Vec& cond) {
    int n = x.size();
    Vec xc(n);
    for (int i = 0; i < n; ++i)
        xc[i] = (!std::isnan(cond[i]) && cond[i] > 0.5) ? x[i] : 0.0;
    return ts_sum(xc, window);
}

Vec ts_lowday(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        int min_idx = 0;
        double mn = x[i - window + 1];
        for (int j = 1; j < window; ++j) {
            double v = x[i - window + 1 + j];
            if (!std::isnan(v) && v < mn) { mn = v; min_idx = j; }
        }
        out[i] = window - min_idx;
    }
    return out;
}

Vec ts_highday(const Vec& x, int window) {
    int n = x.size();
    Vec out(n, NaN);
    for (int i = window - 1; i < n; ++i) {
        int max_idx = 0;
        double mx = x[i - window + 1];
        for (int j = 1; j < window; ++j) {
            double v = x[i - window + 1 + j];
            if (!std::isnan(v) && v > mx) { mx = v; max_idx = j; }
        }
        out[i] = window - max_idx;
    }
    return out;
}

// element-wise
Vec ew_log(const Vec& x) {
    Vec out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = (x[i] > 0) ? std::log(x[i]) : NaN;
    return out;
}
Vec ew_sign(const Vec& x) {
    Vec out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = std::isnan(x[i]) ? NaN : (x[i] > 0 ? 1.0 : (x[i] < 0 ? -1.0 : 0.0));
    return out;
}
Vec ew_abs(const Vec& x) {
    Vec out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = std::isnan(x[i]) ? NaN : std::abs(x[i]);
    return out;
}

#define EW_BINOP(name, op) \
Vec name(const Vec& a, const Vec& b) { \
    int n = a.size(); Vec out(n); \
    for (int i = 0; i < n; ++i) out[i] = (std::isnan(a[i])||std::isnan(b[i])) ? NaN : (a[i] op b[i]); \
    return out; }

EW_BINOP(ew_add, +)
EW_BINOP(ew_sub, -)
EW_BINOP(ew_mul, *)

Vec ew_div(const Vec& a, const Vec& b) {
    int n = a.size(); Vec out(n);
    for (int i = 0; i < n; ++i)
        out[i] = (std::isnan(a[i])||std::isnan(b[i])) ? NaN : safe_div(a[i], b[i]);
    return out;
}
Vec ew_pow(const Vec& a, const Vec& b) {
    int n = a.size(); Vec out(n);
    for (int i = 0; i < n; ++i)
        out[i] = (std::isnan(a[i])||std::isnan(b[i])) ? NaN : std::pow(a[i], b[i]);
    return out;
}
Vec ew_pow_scalar(const Vec& a, double b) {
    Vec out(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        out[i] = std::isnan(a[i]) ? NaN : std::pow(a[i], b);
    return out;
}
Vec ew_scalar_add(const Vec& a, double s) {
    Vec out(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        out[i] = std::isnan(a[i]) ? NaN : a[i] + s;
    return out;
}
Vec ew_scalar_mul(const Vec& a, double s) {
    Vec out(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        out[i] = std::isnan(a[i]) ? NaN : a[i] * s;
    return out;
}
Vec ew_scalar_div(double s, const Vec& a) {
    Vec out(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        out[i] = std::isnan(a[i]) ? NaN : safe_div(s, a[i]);
    return out;
}
Vec ew_max2(const Vec& a, const Vec& b) {
    int n = a.size(); Vec out(n);
    for (int i = 0; i < n; ++i) {
        if (std::isnan(a[i])) out[i] = b[i];
        else if (std::isnan(b[i])) out[i] = a[i];
        else out[i] = std::max(a[i], b[i]);
    }
    return out;
}
Vec ew_min2(const Vec& a, const Vec& b) {
    int n = a.size(); Vec out(n);
    for (int i = 0; i < n; ++i) {
        if (std::isnan(a[i])) out[i] = b[i];
        else if (std::isnan(b[i])) out[i] = a[i];
        else out[i] = std::min(a[i], b[i]);
    }
    return out;
}

// ===========================================================================
//  CROSS-SECTIONAL RANK  (across tickers, per date)
//  Mat layout: [ticker][date]  →  iterate over date index d
// ===========================================================================
Mat cs_rank(const Mat& m) {
    if (m.empty()) return {};
    int T = m.size();          // tickers
    int D = m[0].size();       // dates
    Mat out(T, Vec(D, NaN));
    for (int d = 0; d < D; ++d) {
        // collect non-NaN values
        std::vector<std::pair<double,int>> vals;
        vals.reserve(T);
        for (int t = 0; t < T; ++t) {
            if (!std::isnan(m[t][d])) vals.push_back({m[t][d], t});
        }
        int n = vals.size();
        if (n == 0) continue;
        std::sort(vals.begin(), vals.end());
        // min rank, pct
        int i = 0;
        while (i < n) {
            int j = i;
            while (j < n && vals[j].first == vals[i].first) ++j;
            double rank = (i + 1);   // min rank
            double pct  = rank / n;
            for (int k = i; k < j; ++k)
                out[vals[k].second][d] = pct;
            i = j;
        }
    }
    return out;
}

// ===========================================================================
//  Alpha191  — constructor & helpers
// ===========================================================================

Alpha191::Alpha191(
    const Mat& open, const Mat& high, const Mat& low, const Mat& close,
    const Mat& volume, const Mat& amount, const Mat& returns)
    : open_(open), high_(high), low_(low), close_(close),
      volume_(volume), amount_(amount), returns_(returns)
{
    // vwap = amount / volume
    vwap_ = mat_apply2(amount_, volume_, [](const Vec& a, const Vec& v) {
        return ew_div(a, v);
    });
}

// ---------------------------------------------------------------------------
//  Internal helpers
// ---------------------------------------------------------------------------
Mat Alpha191::mat_apply1(const Mat& a,
                         std::function<Vec(const Vec&)> fn) const {
    Mat out(a.size());
    for (size_t t = 0; t < a.size(); ++t) out[t] = fn(a[t]);
    return out;
}
Mat Alpha191::mat_apply2(const Mat& a, const Mat& b,
                         std::function<Vec(const Vec&,const Vec&)> fn) const {
    Mat out(a.size());
    for (size_t t = 0; t < a.size(); ++t) out[t] = fn(a[t], b[t]);
    return out;
}

Mat Alpha191::zeros() const {
    return Mat(close_.size(), Vec(close_[0].size(), 0.0));
}
Mat Alpha191::scalar_mat(double v) const {
    return Mat(close_.size(), Vec(close_[0].size(), v));
}

// -- Matrix wrappers of 1-D primitives ---------------------------------------
#define MAT1(fn_name, prim) \
Mat Alpha191::fn_name(const Mat& x, int w) const { \
    return mat_apply1(x, [w](const Vec& v){ return prim(v, w); }); }

MAT1(Delay,      ts_delay)
MAT1(Delta,      ts_delta)
MAT1(Sum,        ts_sum)
MAT1(Mean,       ts_mean)
MAT1(Std,        ts_std)
MAT1(Tsmax,      ts_max)
MAT1(Tsmin,      ts_min)
MAT1(Tsrank,     ts_rank)
MAT1(Decaylinear,ts_decaylinear)
MAT1(Lowday,     ts_lowday)
MAT1(Highday,    ts_highday)
MAT1(Count,      ts_count)

Mat Alpha191::Corr(const Mat& x, const Mat& y, int w) const {
    return mat_apply2(x, y, [w](const Vec& a, const Vec& b){ return ts_corr(a,b,w); });
}
Mat Alpha191::Cov(const Mat& x, const Mat& y, int w) const {
    return mat_apply2(x, y, [w](const Vec& a, const Vec& b){ return ts_cov(a,b,w); });
}
Mat Alpha191::Sma(const Mat& x, int n, int m) const {
    return mat_apply1(x, [n,m](const Vec& v){ return ts_sma(v,n,m); });
}
Mat Alpha191::Wma(const Mat& x, int window) const {
    return mat_apply1(x, [window](const Vec& v){ return ts_wma(v,window); });
}
Mat Alpha191::Regbeta(const Mat& y, int window) const {
    return mat_apply1(y, [window](const Vec& v){ return ts_regbeta(v,window); });
}
Mat Alpha191::Sumif(const Mat& x, int window, const Mat& cond) const {
    return mat_apply2(x, cond, [window](const Vec& a, const Vec& c){
        return ts_sumif(a, window, c);
    });
}
Mat Alpha191::Log(const Mat& x) const {
    return mat_apply1(x, [](const Vec& v){ return ew_log(v); });
}
Mat Alpha191::Sign(const Mat& x) const {
    return mat_apply1(x, [](const Vec& v){ return ew_sign(v); });
}
Mat Alpha191::Abs(const Mat& x) const {
    return mat_apply1(x, [](const Vec& v){ return ew_abs(v); });
}

// element-wise matrix ops
#define MAT_BINOP(fname, ew_fn) \
Mat Alpha191::fname(const Mat& a, const Mat& b) const { \
    return mat_apply2(a, b, [](const Vec& x, const Vec& y){ return ew_fn(x,y); }); }

MAT_BINOP(Add, ew_add)
MAT_BINOP(Sub, ew_sub)
MAT_BINOP(Mul, ew_mul)
MAT_BINOP(Div, ew_div)
MAT_BINOP(Pow, ew_pow)
MAT_BINOP(EwMax, ew_max2)
MAT_BINOP(EwMin, ew_min2)

Mat Alpha191::ScalarAdd(const Mat& a, double s) const {
    return mat_apply1(a, [s](const Vec& v){ return ew_scalar_add(v,s); });
}
Mat Alpha191::ScalarMul(const Mat& a, double s) const {
    return mat_apply1(a, [s](const Vec& v){ return ew_scalar_mul(v,s); });
}
Mat Alpha191::ScalarDiv(double s, const Mat& a) const {
    return mat_apply1(a, [s](const Vec& v){ return ew_scalar_div(s,v); });
}
Mat Alpha191::ScalarMax(const Mat& a, double s) const {
    return mat_apply1(a, [s](const Vec& v){
        Vec out(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            out[i] = std::isnan(v[i]) ? NaN : std::max(v[i], s);
        return out;
    });
}
Mat Alpha191::ScalarMin(const Mat& a, double s) const {
    return mat_apply1(a, [s](const Vec& v){
        Vec out(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            out[i] = std::isnan(v[i]) ? NaN : std::min(v[i], s);
        return out;
    });
}
Mat Alpha191::Rank(const Mat& x) const { return cs_rank(x); }

// Where helpers
Mat Alpha191::Where(const Mat& cond, const Mat& a, const Mat& b) const {
    int T = cond.size(), D = cond[0].size();
    Mat out(T, Vec(D));
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            out[t][d] = (!std::isnan(cond[t][d]) && cond[t][d] > 0.5) ? a[t][d] : b[t][d];
    return out;
}
Mat Alpha191::WhereScalar(const Mat& cond, double a, double b) const {
    int T = cond.size(), D = cond[0].size();
    Mat out(T, Vec(D));
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            out[t][d] = (!std::isnan(cond[t][d]) && cond[t][d] > 0.5) ? a : b;
    return out;
}
Mat Alpha191::WhereLeft(const Mat& cond, const Mat& a, double b) const {
    int T = cond.size(), D = cond[0].size();
    Mat out(T, Vec(D));
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            out[t][d] = (!std::isnan(cond[t][d]) && cond[t][d] > 0.5) ? a[t][d] : b;
    return out;
}
Mat Alpha191::WhereRight(const Mat& cond, double a, const Mat& b) const {
    int T = cond.size(), D = cond[0].size();
    Mat out(T, Vec(D));
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            out[t][d] = (!std::isnan(cond[t][d]) && cond[t][d] > 0.5) ? a : b[t][d];
    return out;
}

// boolean comparisons → Mat of 0/1
static Mat mat_gt(const Mat& a, const Mat& b) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,NaN));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        if (!std::isnan(a[t][d])&&!std::isnan(b[t][d])) out[t][d]=(a[t][d]>b[t][d])?1.0:0.0;
    return out;
}
static Mat mat_lt(const Mat& a, const Mat& b) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,NaN));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        if (!std::isnan(a[t][d])&&!std::isnan(b[t][d])) out[t][d]=(a[t][d]<b[t][d])?1.0:0.0;
    return out;
}
static Mat mat_eq(const Mat& a, const Mat& b) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,NaN));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        if (!std::isnan(a[t][d])&&!std::isnan(b[t][d])) out[t][d]=(a[t][d]==b[t][d])?1.0:0.0;
    return out;
}
static Mat mat_ge(const Mat& a, const Mat& b) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,NaN));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        if (!std::isnan(a[t][d])&&!std::isnan(b[t][d])) out[t][d]=(a[t][d]>=b[t][d])?1.0:0.0;
    return out;
}
static Mat mat_le(const Mat& a, const Mat& b) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,NaN));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        if (!std::isnan(a[t][d])&&!std::isnan(b[t][d])) out[t][d]=(a[t][d]<=b[t][d])?1.0:0.0;
    return out;
}
static Mat mat_and(const Mat& a, const Mat& b) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,0.0));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        out[t][d]=((a[t][d]>0.5)&&(b[t][d]>0.5))?1.0:0.0;
    return out;
}
static Mat mat_not(const Mat& a) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,0.0));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        out[t][d]=(a[t][d]<=0.5)?1.0:0.0;
    return out;
}
static Mat mat_scalar_gt(const Mat& a, double s) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,NaN));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        if (!std::isnan(a[t][d])) out[t][d]=(a[t][d]>s)?1.0:0.0;
    return out;
}
static Mat mat_scalar_lt(const Mat& a, double s) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,NaN));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        if (!std::isnan(a[t][d])) out[t][d]=(a[t][d]<s)?1.0:0.0;
    return out;
}
static Mat mat_scalar_le(const Mat& a, double s) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,NaN));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        if (!std::isnan(a[t][d])) out[t][d]=(a[t][d]<=s)?1.0:0.0;
    return out;
}
static Mat mat_scalar_ge(const Mat& a, double s) {
    int T=a.size(), D=a[0].size(); Mat out(T, Vec(D,NaN));
    for (int t=0;t<T;++t) for (int d=0;d<D;++d)
        if (!std::isnan(a[t][d])) out[t][d]=(a[t][d]>=s)?1.0:0.0;
    return out;
}

// ===========================================================================
//  ALPHA IMPLEMENTATIONS  (191 alphas)
// ===========================================================================

Mat Alpha191::alpha001() const {
    // (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6))
    return ScalarMul(Corr(Rank(Delta(Log(volume_),1)), Rank(Div(Sub(close_,open_),open_)), 6), -1.0);
}

Mat Alpha191::alpha002() const {
    // -1 * DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW), 1)
    auto hl = Sub(high_, low_);
    auto num = Sub(Sub(close_,low_), Sub(high_,close_));
    return ScalarMul(Delta(Div(num, hl), 1), -1.0);
}

Mat Alpha191::alpha003() const {
    auto d1 = Delay(close_,1);
    auto cond_eq  = mat_eq(close_, d1);
    auto cond_gt  = mat_gt(close_, d1);
    auto cond_lt  = mat_lt(close_, d1);
    // when gt: close - min(low, delay(close,1))
    auto branch_gt = Sub(close_, EwMin(low_, d1));
    // when lt: close - max(high, delay(close,1))
    auto branch_lt = Sub(close_, EwMax(high_, d1));
    // assemble: eq→0, gt→branch_gt, lt→branch_lt
    auto part = Where(cond_eq, zeros(), Where(cond_gt, branch_gt, branch_lt));
    return Sum(part, 6);
}

Mat Alpha191::alpha004() const {
    auto mean8  = Mean(close_,8);
    auto std8   = Std(close_,8);
    auto mean2  = Mean(close_,2);
    auto thresh = Add(mean8, std8);
    auto vol_ratio = Div(volume_, Mean(volume_,20));
    auto cond1 = mat_lt(thresh, mean2);        // thresh < mean2/2? — original formula: thresh < sum/2
    // re-read: sum(close,2)/2 = Mean(close,2)
    auto cond2 = mat_gt(thresh, mean2);
    auto cond3_base = mat_eq(thresh, mean2);
    auto cond4  = mat_scalar_ge(vol_ratio, 1.0);
    // cond3 & cond4 → 1, else cond3 → -1
    auto cond3_and_4 = mat_and(cond3_base, cond4);
    // cond1 → -1, cond2 → 1, cond3 → -1, cond3&cond4 → 1
    Mat part = WhereScalar(cond1, -1.0, 0.0);
    int T=part.size(), D=part[0].size();
    for (int t=0;t<T;++t) for (int d=0;d<D;++d) {
        if (cond2[t][d]>0.5) part[t][d]=1.0;
        else if (cond3_and_4[t][d]>0.5) part[t][d]=1.0;
        else if (cond3_base[t][d]>0.5) part[t][d]=-1.0;
    }
    return part;
}

Mat Alpha191::alpha005() const {
    // -1 * TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3)
    return ScalarMul(Tsmax(Corr(Tsrank(volume_,5), Tsrank(high_,5), 5), 3), -1.0);
}

Mat Alpha191::alpha006() const {
    // RANK(SIGN(DELTA((OPEN*0.85 + HIGH*0.15), 4))) * -1
    auto combo = Add(ScalarMul(open_,0.85), ScalarMul(high_,0.15));
    return ScalarMul(Rank(Sign(Delta(combo,4))), -1.0);
}

Mat Alpha191::alpha007() const {
    auto diff = Sub(vwap_, close_);
    return Mul(Add(Rank(Tsmax(diff,3)), Rank(Tsmin(diff,3))), Rank(Delta(volume_,3)));
}

Mat Alpha191::alpha008() const {
    auto combo = Add(ScalarMul(Add(high_,low_),0.1), ScalarMul(vwap_,0.8));
    return Rank(ScalarMul(Delta(combo,4), -1.0));
}

Mat Alpha191::alpha009() const {
    // SMA(((H+L)/2 - (dH+dL)/2)*(H-L)/V, 7, 2)
    auto mid     = ScalarMul(Add(high_,low_), 0.5);
    auto mid_d1  = ScalarMul(Add(Delay(high_,1),Delay(low_,1)), 0.5);
    auto hl      = Sub(high_,low_);
    return Sma(Div(Mul(Sub(mid,mid_d1), hl), volume_), 7, 2);
}

Mat Alpha191::alpha010() const {
    auto cond_neg = mat_lt(returns_, zeros());
    auto std20    = Std(returns_,20);
    auto branch   = Where(cond_neg, std20, close_);
    return Rank(Tsmax(Mul(branch, branch), 5));
}

Mat Alpha191::alpha011() const {
    auto num = Sub(Sub(close_,low_), Sub(high_,close_));
    return Sum(Div(Mul(num, volume_), Sub(high_,low_)), 6);
}

Mat Alpha191::alpha012() const {
    auto vwap10 = ScalarMul(Sum(vwap_,10), 1.0/10);
    return ScalarMul(Mul(Rank(Sub(open_, vwap10)), Rank(Abs(Sub(close_, vwap_)))), -1.0);
}

Mat Alpha191::alpha013() const {
    // element-wise sqrt of high*low
    return Sub(mat_apply1(Mul(high_,low_), [](const Vec& v){
        Vec out(v.size());
        for (size_t i=0;i<v.size();++i) out[i]=std::isnan(v[i])?NaN:std::sqrt(std::max(0.0,v[i]));
        return out;
    }), vwap_);
}

Mat Alpha191::alpha014() const {
    return Sub(close_, Delay(close_,5));
}

Mat Alpha191::alpha015() const {
    return ScalarAdd(Div(open_, Delay(close_,1)), -1.0);
}

Mat Alpha191::alpha016() const {
    return ScalarMul(Tsmax(Rank(Corr(Rank(volume_), Rank(vwap_), 5)), 5), -1.0);
}

Mat Alpha191::alpha017() const {
    return Pow(Rank(Sub(vwap_, Tsmax(vwap_,15))), Delta(close_,5));
}

Mat Alpha191::alpha018() const {
    return Div(close_, Delay(close_,5));
}

Mat Alpha191::alpha019() const {
    auto d5   = Delay(close_,5);
    auto gt   = mat_gt(close_, d5);
    auto eq   = mat_eq(close_, d5);
    auto lt   = mat_lt(close_, d5);
    auto ret5 = Div(Sub(close_,d5), d5);
    auto ret5c= Div(Sub(close_,d5), close_);
    return Where(eq, zeros(), Where(lt, ret5, ret5c));
}

Mat Alpha191::alpha020() const {
    return ScalarMul(Div(Sub(close_, Delay(close_,6)), Delay(close_,6)), 100.0);
}

Mat Alpha191::alpha021() const {
    return Regbeta(Mean(close_,6), 6);
}

Mat Alpha191::alpha022() const {
    auto base = Div(Sub(close_, Mean(close_,6)), Mean(close_,6));
    return Sma(Sub(base, Delay(base,3)), 12, 1);
}

Mat Alpha191::alpha023() const {
    auto cond = mat_gt(close_, Delay(close_,1));
    auto std20 = Std(close_,20);
    auto up   = WhereLeft(cond, std20, 0.0);
    auto dn   = WhereRight(cond, 0.0, std20);
    auto su   = Sma(up,20,1);
    auto sd   = Sma(dn,20,1);
    return ScalarMul(Div(su, Add(su,sd)), 100.0);
}

Mat Alpha191::alpha024() const {
    return Sma(Sub(close_, Delay(close_,5)), 5, 1);
}

Mat Alpha191::alpha025() const {
    auto term1 = ScalarMul(Rank(Mul(Delta(close_,7), Sub(scalar_mat(1.0), Rank(Decaylinear(Div(volume_, Mean(volume_,20)),9))))), -1.0);
    auto term2 = ScalarAdd(Rank(Sum(returns_,250)), 1.0);
    return Mul(term1, term2);
}

Mat Alpha191::alpha026() const {
    auto ma7   = ScalarMul(Sum(close_,7), 1.0/7);
    auto corr  = Corr(vwap_, Delay(close_,5), 230);
    return Add(Sub(ma7, close_), corr);
}

Mat Alpha191::alpha027() const {
    auto A = Add(ScalarMul(Div(Sub(close_,Delay(close_,3)),Delay(close_,3)),100.0),
                 ScalarMul(Div(Sub(close_,Delay(close_,6)),Delay(close_,6)),100.0));
    return Wma(A,12);
}

Mat Alpha191::alpha028() const {
    auto ts_lo9 = Tsmin(low_,9);
    auto ts_hi9 = Tsmax(high_,9);
    auto range9 = Sub(ts_hi9, ts_lo9);
    auto stoch  = ScalarMul(Div(Sub(close_,ts_lo9), range9), 100.0);
    auto p1 = ScalarMul(Sma(stoch,3,1), 3.0);
    auto p2 = ScalarMul(Sma(Sma(stoch,3,1),3,1), 2.0);
    return Sub(p1,p2);
}

Mat Alpha191::alpha029() const {
    return Mul(Div(Sub(close_,Delay(close_,6)),Delay(close_,6)), volume_);
}

Mat Alpha191::alpha030() const { return zeros(); }  // requires external market data

Mat Alpha191::alpha031() const {
    return ScalarMul(Div(Sub(close_,Mean(close_,12)),Mean(close_,12)), 100.0);
}

Mat Alpha191::alpha032() const {
    return ScalarMul(Sum(Rank(Corr(Rank(high_),Rank(volume_),3)),3), -1.0);
}

Mat Alpha191::alpha033() const {
    auto term1 = Add(ScalarMul(Tsmin(low_,5),-1.0), Delay(Tsmin(low_,5),5));
    auto term2 = Rank(ScalarMul(Div(Sub(Sum(returns_,240),Sum(returns_,20)),scalar_mat(220.0)),1.0));
    return Mul(Mul(term1,term2), Tsrank(volume_,5));
}

Mat Alpha191::alpha034() const {
    return Div(Mean(close_,12), close_);
}

Mat Alpha191::alpha035() const {
    auto A = Rank(Decaylinear(Delta(open_,1),15));
    auto B = Rank(Decaylinear(Corr(volume_,Add(ScalarMul(open_,0.65),ScalarMul(open_,0.35)),17),7));
    return ScalarMul(EwMin(A,B), -1.0);
}

Mat Alpha191::alpha036() const {
    return Rank(Sum(Corr(Rank(volume_),Rank(vwap_),6),2));
}

Mat Alpha191::alpha037() const {
    auto prod = Mul(Sum(open_,5),Sum(returns_,5));
    return ScalarMul(Rank(Sub(prod,Delay(prod,10))), -1.0);
}

Mat Alpha191::alpha038() const {
    auto cond = mat_gt(Mean(high_,20), high_);    // mean < high → negate delta; else 0
    // original: if (sum(high,20)/20 < high) → -delta; else 0
    // cond_lt means sum/20 < high
    auto cond_lt = mat_lt(Mean(high_,20), high_);
    return WhereLeft(cond_lt, ScalarMul(Delta(high_,2),-1.0), 0.0);
}

Mat Alpha191::alpha039() const {
    auto combo = Add(ScalarMul(vwap_,0.3),ScalarMul(open_,0.7));
    auto A = Rank(Decaylinear(Delta(close_,2),8));
    auto B = Rank(Decaylinear(Corr(combo, Sum(Mean(volume_,180),37), 14), 12));
    return ScalarMul(Sub(A,B), -1.0);
}

Mat Alpha191::alpha040() const {
    auto cond = mat_gt(close_, Delay(close_,1));
    auto up   = WhereLeft(cond, volume_, 0.0);
    auto dn   = WhereRight(cond, 0.0, volume_);
    return ScalarMul(Div(Sum(up,26),Sum(dn,26)), 100.0);
}

Mat Alpha191::alpha041() const {
    return ScalarMul(Rank(Tsmax(Delta(vwap_,3),5)), -1.0);
}

Mat Alpha191::alpha042() const {
    return Mul(ScalarMul(Rank(Std(high_,10)),-1.0), Corr(high_,volume_,10));
}

Mat Alpha191::alpha043() const {
    auto gt = mat_gt(close_,Delay(close_,1));
    auto lt = mat_lt(close_,Delay(close_,1));
    auto eq = mat_eq(close_,Delay(close_,1));
    auto part = Where(gt, volume_, Where(lt, ScalarMul(volume_,-1.0), zeros()));
    return Sum(part,6);
}

Mat Alpha191::alpha044() const {
    auto A = Tsrank(Decaylinear(Corr(low_,Mean(volume_,10),7),6),4);
    auto B = Tsrank(Decaylinear(Delta(vwap_,3),10),15);
    return Add(A,B);
}

Mat Alpha191::alpha045() const {
    auto combo = Add(ScalarMul(close_,0.6),ScalarMul(open_,0.4));
    return Mul(Rank(Delta(combo,1)), Rank(Corr(vwap_,Mean(volume_,150),15)));
}

Mat Alpha191::alpha046() const {
    auto s = Add(Add(Mean(close_,3),Mean(close_,6)),Add(Mean(close_,12),Mean(close_,24)));
    return Div(s, ScalarMul(close_,4.0));
}

Mat Alpha191::alpha047() const {
    auto hi6 = Tsmax(high_,6);
    auto lo6 = Tsmin(low_,6);
    return Sma(ScalarMul(Div(Sub(hi6,close_),Sub(hi6,lo6)),100.0),9,1);
}

Mat Alpha191::alpha048() const {
    auto s1 = Sign(Sub(close_,Delay(close_,1)));
    auto s2 = Sign(Sub(Delay(close_,1),Delay(close_,2)));
    auto s3 = Sign(Sub(Delay(close_,2),Delay(close_,3)));
    auto rank_sum = Rank(Add(Add(s1,s2),s3));
    return ScalarMul(Div(Mul(rank_sum,Sum(volume_,5)),Sum(volume_,20)), -1.0);
}

Mat Alpha191::alpha049() const {
    auto hl  = Add(high_,low_);
    auto dhl = Add(Delay(high_,1),Delay(low_,1));
    auto cond = mat_gt(hl, dhl);   // if hl > dhl → 0, else max(...)
    auto mx   = EwMax(Abs(Sub(high_,Delay(high_,1))), Abs(Sub(low_,Delay(low_,1))));
    auto part1 = WhereRight(cond, 0.0, mx);  // 0 if gt, mx if not
    auto part2 = WhereLeft(cond,  mx, 0.0);  // mx if gt, 0 if not
    return Div(Sum(part1,12), Add(Sum(part1,12),Sum(part2,12)));
}

Mat Alpha191::alpha050() const {
    auto hl  = Add(high_,low_);
    auto dhl = Add(Delay(high_,1),Delay(low_,1));
    auto cond = mat_le(hl, dhl);   // if hl <= dhl → 0 in part1, mx in part2
    auto mx   = EwMax(Abs(Sub(high_,Delay(high_,1))), Abs(Sub(low_,Delay(low_,1))));
    auto part1 = WhereRight(cond, 0.0, mx);
    auto part2 = WhereLeft(cond, mx, 0.0);
    auto s1 = Sum(part1,12), s2 = Sum(part2,12);
    return Div(Sub(s1,s2), Add(s1,s2));
}

Mat Alpha191::alpha051() const { return alpha049(); }

Mat Alpha191::alpha052() const {
    auto tp_d1 = Delay(ScalarMul(Add(Add(high_,low_),close_), 1.0/3), 1);
    return ScalarMul(Div(Sum(ScalarMax(Sub(high_,tp_d1),0.0),26),
                         Sum(ScalarMax(Sub(tp_d1,low_),0.0),26)), 100.0);
}

Mat Alpha191::alpha053() const {
    auto cond = mat_gt(close_,Delay(close_,1));
    return ScalarMul(Div(Count(cond,12),scalar_mat(12.0)),100.0);
}

Mat Alpha191::alpha054() const {
    // (-1 * RANK((STD(ABS(CLOSE-OPEN))+(CLOSE-OPEN))+CORR(CLOSE,OPEN,10)))
    // Note: original uses cross-sectional std of abs(close-open) — we use ts_std
    auto diff = Sub(close_,open_);
    auto inner = Add(Add(Std(Abs(diff),10), diff), Corr(close_,open_,10));
    return ScalarMul(Rank(inner), -1.0);
}

Mat Alpha191::alpha055() const {
    auto A = Abs(Sub(high_, Delay(close_,1)));
    auto B = Abs(Sub(low_,  Delay(close_,1)));
    auto C = Abs(Sub(high_, Delay(low_,1)));
    auto D = Abs(Sub(Delay(close_,1), Delay(open_,1)));
    auto condA = mat_and(mat_gt(A,B), mat_gt(A,C));
    auto condB = mat_and(mat_gt(B,C), mat_gt(B,A));
    auto part0 = ScalarMul(Add(Sub(close_, ScalarMul(Sub(close_,open_),0.5)), Delay(open_,1)), 16.0);  // simplified
    // Actually: 16*(close + (close-open)/2 - delay(open,1))
    auto p0 = ScalarMul(Sub(Add(close_, ScalarMul(Sub(close_,open_),0.5)), Delay(open_,1)), 16.0);
    auto brA = Add(Add(A, ScalarMul(B,0.5)), ScalarMul(D,0.25));
    auto brB = Add(Add(B, ScalarMul(A,0.5)), ScalarMul(D,0.25));
    auto brC = Add(C, ScalarMul(D,0.25));
    auto part1 = Where(condA, brA, Where(condB, brB, brC));
    auto part2 = EwMax(A,B);
    return Sum(Div(Mul(p0,part2),part1),20);
}

Mat Alpha191::alpha056() const {
    auto A = Rank(Sub(open_, Tsmin(open_,12)));
    auto sum_mid = Sum(ScalarMul(Add(high_,low_),0.5),19);
    auto sum_vol = Sum(Mean(volume_,40),19);
    auto B = Rank(Pow(Rank(Corr(sum_mid,sum_vol,13)),scalar_mat(5.0)));
    return mat_lt(A,B);   // 1 if A<B, 0 otherwise — already 0/1
}

Mat Alpha191::alpha057() const {
    auto hi9=Tsmax(high_,9), lo9=Tsmin(low_,9);
    return Sma(ScalarMul(Div(Sub(close_,lo9),Sub(hi9,lo9)),100.0),3,1);
}

Mat Alpha191::alpha058() const {
    auto cond = mat_gt(close_,Delay(close_,1));
    return ScalarMul(Div(Count(cond,20),scalar_mat(20.0)),100.0);
}

Mat Alpha191::alpha059() const {
    auto d1=Delay(close_,1);
    auto eq=mat_eq(close_,d1), gt=mat_gt(close_,d1), lt=mat_lt(close_,d1);
    auto brGt=Sub(close_,EwMin(low_,d1));
    auto brLt=Sub(close_,EwMax(low_,d1));
    return Sum(Where(eq,zeros(),Where(gt,brGt,brLt)),20);
}

Mat Alpha191::alpha060() const {
    auto num=Sub(Sub(close_,low_),Sub(high_,close_));
    return Sum(Div(Mul(num,volume_),Sub(high_,low_)),20);
}

Mat Alpha191::alpha061() const {
    auto A=Rank(Decaylinear(Delta(vwap_,1),12));
    auto B=Rank(Decaylinear(Rank(Corr(low_,Mean(volume_,80),8)),17));
    return ScalarMul(EwMax(A,B),-1.0);
}

Mat Alpha191::alpha062() const {
    return ScalarMul(Corr(high_,Rank(volume_),5),-1.0);
}

Mat Alpha191::alpha063() const {
    auto diff=Sub(close_,Delay(close_,1));
    return ScalarMul(Div(Sma(ScalarMax(diff,0.0),6,1),Sma(Abs(diff),6,1)),100.0);
}

Mat Alpha191::alpha064() const {
    auto A=Rank(Decaylinear(Corr(Rank(vwap_),Rank(volume_),4),4));
    auto B=Rank(Decaylinear(Tsmax(Corr(Rank(close_),Rank(Mean(volume_,60)),4),13),14));
    return ScalarMul(EwMax(A,B),-1.0);
}

Mat Alpha191::alpha065() const { return Div(Mean(close_,6),close_); }

Mat Alpha191::alpha066() const {
    return ScalarMul(Div(Sub(close_,Mean(close_,6)),Mean(close_,6)),100.0);
}

Mat Alpha191::alpha067() const {
    auto diff=Sub(close_,Delay(close_,1));
    return ScalarMul(Div(Sma(ScalarMax(diff,0.0),24,1),Sma(Abs(diff),24,1)),100.0);
}

Mat Alpha191::alpha068() const {
    auto mid=ScalarMul(Add(high_,low_),0.5);
    auto mid_d1=ScalarMul(Add(Delay(high_,1),Delay(low_,1)),0.5);
    return Sma(Div(Mul(Sub(mid,mid_d1),Sub(high_,low_)),volume_),15,2);
}

Mat Alpha191::alpha069() const {
    auto cond_open_le=mat_le(open_,Delay(open_,1));
    auto cond_open_ge=mat_ge(open_,Delay(open_,1));
    auto DTM=WhereRight(cond_open_le,0.0,EwMax(Sub(high_,open_),Sub(open_,Delay(open_,1))));
    auto DBM=WhereRight(cond_open_ge,0.0,EwMax(Sub(open_,low_),Sub(open_,Delay(open_,1))));
    auto sDTM=Sum(DTM,20), sDBM=Sum(DBM,20);
    auto cond_gt=mat_gt(sDTM,sDBM);
    auto cond_eq=mat_eq(sDTM,sDBM);
    auto cond_lt=mat_lt(sDTM,sDBM);
    auto brGt=Div(Sub(sDTM,sDBM),sDTM);
    auto brLt=Div(Sub(sDTM,sDBM),sDBM);
    return Where(cond_gt,brGt,Where(cond_eq,zeros(),brLt));
}

Mat Alpha191::alpha070() const { return Std(amount_,6); }

Mat Alpha191::alpha071() const {
    return ScalarMul(Div(Sub(close_,Mean(close_,24)),Mean(close_,24)),100.0);
}

Mat Alpha191::alpha072() const {
    auto hi6=Tsmax(high_,6), lo6=Tsmin(low_,6);
    return Sma(ScalarMul(Div(Sub(hi6,close_),Sub(hi6,lo6)),100.0),15,1);
}

Mat Alpha191::alpha073() const {
    auto A=Tsrank(Decaylinear(Decaylinear(Corr(close_,volume_,10),16),4),5);
    auto B=Rank(Decaylinear(Corr(vwap_,Mean(volume_,30),4),3));
    return ScalarMul(Sub(A,B),-1.0);
}

Mat Alpha191::alpha074() const {
    auto combo=Add(ScalarMul(low_,0.35),ScalarMul(vwap_,0.65));
    auto A=Rank(Corr(Sum(combo,20),Sum(Mean(volume_,40),20),7));
    auto B=Rank(Corr(Rank(vwap_),Rank(volume_),6));
    return Add(A,B);
}

Mat Alpha191::alpha075() const { return zeros(); }  // requires benchmark

Mat Alpha191::alpha076() const {
    auto ret=Abs(Div(Sub(close_,Delay(close_,1)),Delay(close_,1)));
    auto r_v=Div(ret,volume_);
    return Div(Std(r_v,20),Mean(r_v,20));
}

Mat Alpha191::alpha077() const {
    auto mid_plus_high=Add(ScalarMul(Add(high_,low_),0.5),high_);
    auto A=Rank(Decaylinear(Sub(mid_plus_high,Add(vwap_,high_)),20));
    auto B=Rank(Decaylinear(Corr(ScalarMul(Add(high_,low_),0.5),Mean(volume_,40),3),6));
    return EwMin(A,B);
}

Mat Alpha191::alpha078() const {
    auto tp=ScalarMul(Add(Add(high_,low_),close_),1.0/3);
    auto mean_tp=Mean(tp,12);
    auto md=Mean(Abs(Sub(close_,mean_tp)),12);
    return Div(Sub(tp,mean_tp),ScalarMul(md,0.015));
}

Mat Alpha191::alpha079() const {
    auto diff=Sub(close_,Delay(close_,1));
    return ScalarMul(Div(Sma(ScalarMax(diff,0.0),12,1),Sma(Abs(diff),12,1)),100.0);
}

Mat Alpha191::alpha080() const {
    return ScalarMul(Div(Sub(volume_,Delay(volume_,5)),Delay(volume_,5)),100.0);
}

Mat Alpha191::alpha081() const { return Sma(volume_,21,2); }

Mat Alpha191::alpha082() const {
    auto hi6=Tsmax(high_,6), lo6=Tsmin(low_,6);
    return Sma(ScalarMul(Div(Sub(hi6,close_),Sub(hi6,lo6)),100.0),20,1);
}

Mat Alpha191::alpha083() const {
    return ScalarMul(Rank(Cov(Rank(high_),Rank(volume_),5)),-1.0);
}

Mat Alpha191::alpha084() const {
    auto gt=mat_gt(close_,Delay(close_,1)), lt=mat_lt(close_,Delay(close_,1));
    auto eq=mat_eq(close_,Delay(close_,1));
    auto part=Where(gt,volume_,Where(lt,zeros(),ScalarMul(volume_,-1.0)));
    return Sum(part,20);
}

Mat Alpha191::alpha085() const {
    return Mul(Tsrank(Div(volume_,Mean(volume_,20)),20),
               Tsrank(ScalarMul(Delta(close_,7),-1.0),8));
}

Mat Alpha191::alpha086() const {
    auto A=Div(Sub(Delay(close_,20),Delay(close_,10)),scalar_mat(10.0));
    auto B=Div(Sub(Delay(close_,10),close_),scalar_mat(10.0));
    auto acc=Sub(A,B);
    auto cond_gt=mat_scalar_gt(acc,0.25);
    auto cond_lt=mat_scalar_lt(acc,0.0);
    return Where(cond_gt,scalar_mat(-1.0),
           Where(cond_lt,scalar_mat(1.0),
           ScalarMul(Sub(close_,Delay(close_,1)),-1.0)));
}

Mat Alpha191::alpha087() const {
    auto A=Rank(Decaylinear(Delta(vwap_,4),7));
    auto hl2=ScalarMul(Add(high_,low_),0.5);
    auto num=Sub(Sub(ScalarMul(low_,0.9),vwap_),Sub(open_,hl2));  // simplified; matches Python
    // Python: (low*0.9 + low*0.1 - vwap)/(open - (high+low)/2)  → low - vwap / ...
    auto low_mix=Add(ScalarMul(low_,0.9),ScalarMul(low_,0.1));  // = low
    auto B=Tsrank(Decaylinear(Div(Sub(low_mix,vwap_),Sub(open_,hl2)),11),7);
    return ScalarMul(Add(A,B),-1.0);
}

Mat Alpha191::alpha088() const {
    return ScalarMul(Div(Sub(close_,Delay(close_,20)),Delay(close_,20)),100.0);
}

Mat Alpha191::alpha089() const {
    auto diff=Sub(Sma(close_,13,2),Sma(close_,27,2));
    return ScalarMul(Sub(diff,Sma(diff,10,2)),2.0);
}

Mat Alpha191::alpha090() const {
    return ScalarMul(Rank(Corr(Rank(vwap_),Rank(volume_),5)),-1.0);
}

Mat Alpha191::alpha091() const {
    return ScalarMul(Mul(Rank(Sub(close_,Tsmax(close_,5))),
                        Rank(Corr(Mean(volume_,40),low_,5))),-1.0);
}

Mat Alpha191::alpha092() const {
    auto combo=Add(ScalarMul(close_,0.35),ScalarMul(vwap_,0.65));
    auto A=Rank(Decaylinear(Delta(combo,2),3));
    auto B=Tsrank(Decaylinear(Abs(Corr(Mean(volume_,180),close_,13)),5),15);
    return ScalarMul(EwMax(A,B),-1.0);
}

Mat Alpha191::alpha093() const {
    auto cond=mat_ge(open_,Delay(open_,1));
    auto mx=EwMax(Sub(open_,low_),Sub(open_,Delay(open_,1)));
    return Sum(WhereRight(cond,0.0,mx),20);
}

Mat Alpha191::alpha094() const {
    auto gt=mat_gt(close_,Delay(close_,1)), lt=mat_lt(close_,Delay(close_,1));
    auto part=Where(gt,volume_,Where(lt,ScalarMul(volume_,-1.0),zeros()));
    return Sum(part,30);
}

Mat Alpha191::alpha095() const { return Std(amount_,20); }

Mat Alpha191::alpha096() const {
    auto hi9=Tsmax(high_,9), lo9=Tsmin(low_,9);
    auto stoch=ScalarMul(Div(Sub(close_,lo9),Sub(hi9,lo9)),100.0);
    return Sma(Sma(stoch,3,1),3,1);
}

Mat Alpha191::alpha097() const { return Std(volume_,10); }

Mat Alpha191::alpha098() const {
    auto sma100=Sum(close_,100);
    auto mean100=ScalarMul(sma100,0.01);
    auto acc=Div(Delta(mean100,100),Delay(close_,100));
    auto cond=mat_scalar_le(acc,0.05);
    auto brTrue=ScalarMul(Sub(close_,Tsmin(close_,100)),-1.0);
    auto brFalse=ScalarMul(Delta(close_,3),-1.0);
    return Where(cond,brTrue,brFalse);
}

Mat Alpha191::alpha099() const {
    return ScalarMul(Rank(Cov(Rank(close_),Rank(volume_),5)),-1.0);
}

Mat Alpha191::alpha100() const { return Std(volume_,20); }

Mat Alpha191::alpha101() const {
    auto r1=Rank(Corr(close_,Sum(Mean(volume_,30),37),15));
    auto r2=Rank(Corr(Rank(Add(ScalarMul(high_,0.1),ScalarMul(vwap_,0.9))),Rank(volume_),11));
    return mat_lt(r1,r2);
}

Mat Alpha191::alpha102() const {
    auto dv=Sub(volume_,Delay(volume_,1));
    return ScalarMul(Div(Sma(ScalarMax(dv,0.0),6,1),Sma(Abs(dv),6,1)),100.0);
}

Mat Alpha191::alpha103() const {
    return ScalarMul(Div(Sub(scalar_mat(20.0),Lowday(low_,20)),scalar_mat(20.0)),100.0);
}

Mat Alpha191::alpha104() const {
    return ScalarMul(Mul(Delta(Corr(high_,volume_,5),5),Rank(Std(close_,20))),-1.0);
}

Mat Alpha191::alpha105() const {
    return ScalarMul(Corr(Rank(open_),Rank(volume_),10),-1.0);
}

Mat Alpha191::alpha106() const { return Sub(close_,Delay(close_,20)); }

Mat Alpha191::alpha107() const {
    return Mul(Mul(ScalarMul(Rank(Sub(open_,Delay(high_,1))),-1.0),
                   Rank(Sub(open_,Delay(close_,1)))),
               Rank(Sub(open_,Delay(low_,1))));
}

Mat Alpha191::alpha108() const {
    return ScalarMul(Pow(Rank(Sub(high_,Tsmin(high_,2))),Rank(Corr(vwap_,Mean(volume_,120),6))),-1.0);
}

Mat Alpha191::alpha109() const {
    auto hl=Sub(high_,low_);
    return Div(Sma(hl,10,2),Sma(Sma(hl,10,2),10,2));
}

Mat Alpha191::alpha110() const {
    return ScalarMul(Div(Sum(ScalarMax(Sub(high_,Delay(close_,1)),0.0),20),
                        Sum(ScalarMax(Sub(Delay(close_,1),low_),0.0),20)),100.0);
}

Mat Alpha191::alpha111() const {
    auto clr=Div(Mul(volume_,Sub(Sub(close_,low_),Sub(high_,close_))),Sub(high_,low_));
    return Sub(Sma(clr,11,2),Sma(clr,4,2));
}

Mat Alpha191::alpha112() const {
    auto diff=Sub(close_,Delay(close_,1));
    auto cond=mat_scalar_gt(diff,0.0);
    auto up=WhereLeft(cond,diff,0.0);
    auto dn=WhereRight(cond,0.0,Abs(diff));
    auto su=Sum(up,12), sd=Sum(dn,12);
    return ScalarMul(Div(Sub(su,sd),Add(su,sd)),100.0);
}

Mat Alpha191::alpha113() const {
    auto A=Rank(ScalarMul(Sum(Delay(close_,5),20),0.05));
    return ScalarMul(Mul(Mul(A,Corr(close_,volume_,2)),Rank(Corr(Sum(close_,5),Sum(close_,20),2))),-1.0);
}

Mat Alpha191::alpha114() const {
    auto hl_norm=Div(Sub(high_,low_),ScalarMul(Sum(close_,5),0.2));
    auto A=Rank(Delay(hl_norm,2));
    auto B=Rank(Rank(volume_));
    auto denom=Div(hl_norm,Sub(vwap_,close_));
    return Div(Mul(A,B),denom);
}

Mat Alpha191::alpha115() const {
    auto combo=Add(ScalarMul(high_,0.9),ScalarMul(close_,0.1));
    auto A=Rank(Corr(combo,Mean(volume_,30),10));
    auto B=Rank(Corr(Tsrank(ScalarMul(Add(high_,low_),0.5),4),Tsrank(volume_,10),7));
    return Pow(A,B);
}

Mat Alpha191::alpha116() const { return Regbeta(close_,20); }

Mat Alpha191::alpha117() const {
    auto A=Tsrank(volume_,32);
    auto B=ScalarAdd(Tsrank(Add(Sub(close_,high_),low_),16),-1.0);  // 1 - tsrank(...)  Note: negated
    auto C=ScalarAdd(Tsrank(returns_,32),-1.0);
    // (A*(1-B))*(1-C) — but B and C are tsranks not (1-tsrank)
    return Mul(Mul(A,Sub(scalar_mat(1.0),Tsrank(Sub(Add(close_,high_),low_),16))),
               Sub(scalar_mat(1.0),Tsrank(returns_,32)));
}

Mat Alpha191::alpha118() const {
    return ScalarMul(Div(Sum(Sub(high_,open_),20),Sum(Sub(open_,low_),20)),100.0);
}

Mat Alpha191::alpha119() const {
    auto A=Rank(Decaylinear(Corr(vwap_,Sum(Mean(volume_,5),26),5),7));
    auto B=Rank(Decaylinear(Tsrank(Tsmin(Corr(Rank(open_),Rank(Mean(volume_,15)),21),9),7),8));
    return Sub(A,B);
}

Mat Alpha191::alpha120() const {
    return Div(Rank(Sub(vwap_,close_)),Rank(Add(vwap_,close_)));
}

Mat Alpha191::alpha121() const {
    auto base=Rank(Sub(vwap_,Tsmin(vwap_,12)));
    auto exp_=Tsrank(Corr(Tsrank(vwap_,20),Tsrank(Mean(volume_,60),2),18),3);
    return ScalarMul(Pow(base,exp_),-1.0);
}

Mat Alpha191::alpha122() const {
    auto sss=Sma(Sma(Sma(Log(close_),13,2),13,2),13,2);
    return Div(Sub(sss,Delay(sss,1)),Delay(sss,1));
}

Mat Alpha191::alpha123() const {
    auto A=Rank(Corr(Sum(ScalarMul(Add(high_,low_),0.5),20),Sum(Mean(volume_,60),20),9));
    auto B=Rank(Corr(low_,volume_,6));
    return WhereScalar(mat_lt(A,B),-1.0,0.0);
}

Mat Alpha191::alpha124() const {
    return Div(Sub(close_,vwap_),Decaylinear(Rank(Tsmax(close_,30)),2));
}

Mat Alpha191::alpha125() const {
    auto A=Rank(Decaylinear(Corr(vwap_,Mean(volume_,80),17),20));
    auto combo=Add(ScalarMul(close_,0.5),ScalarMul(vwap_,0.5));
    auto B=Rank(Decaylinear(Delta(combo,3),16));
    return Div(A,B);
}

Mat Alpha191::alpha126() const {
    return ScalarMul(Add(Add(close_,high_),low_),1.0/3);
}

Mat Alpha191::alpha127() const {
    auto pct=ScalarMul(Div(Sub(close_,Tsmax(close_,12)),Tsmax(close_,12)),100.0);
    return mat_apply1(Mean(Mul(pct,pct),12),[](const Vec& v){
        Vec out(v.size());
        for (size_t i=0;i<v.size();++i) out[i]=std::isnan(v[i])?NaN:std::sqrt(v[i]);
        return out;
    });
}

Mat Alpha191::alpha128() const {
    auto tp=ScalarMul(Add(Add(high_,low_),close_),1.0/3);
    auto cond=mat_gt(tp,Delay(tp,1));
    auto up=WhereLeft(cond,Mul(tp,volume_),0.0);
    auto dn=WhereRight(cond,0.0,Mul(tp,volume_));
    return Sub(scalar_mat(100.0),Div(scalar_mat(100.0),ScalarAdd(Div(Sum(up,14),Sum(dn,14)),1.0)));
}

Mat Alpha191::alpha129() const {
    auto diff=Sub(close_,Delay(close_,1));
    auto cond=mat_scalar_lt(diff,0.0);
    return Sum(WhereLeft(cond,Abs(diff),0.0),12);
}

Mat Alpha191::alpha130() const {
    auto A=Rank(Decaylinear(Corr(ScalarMul(Add(high_,low_),0.5),Mean(volume_,40),9),10));
    auto B=Rank(Decaylinear(Corr(Rank(vwap_),Rank(volume_),7),3));
    return Div(A,B);
}

Mat Alpha191::alpha131() const {
    return Pow(Rank(Delta(vwap_,1)),Tsrank(Corr(close_,Mean(volume_,50),18),18));
}

Mat Alpha191::alpha132() const { return Mean(amount_,20); }

Mat Alpha191::alpha133() const {
    return Sub(ScalarMul(Div(Sub(scalar_mat(20.0),Highday(high_,20)),scalar_mat(20.0)),100.0),
               ScalarMul(Div(Sub(scalar_mat(20.0),Lowday(low_,20)),scalar_mat(20.0)),100.0));
}

Mat Alpha191::alpha134() const {
    return Mul(Div(Sub(close_,Delay(close_,12)),Delay(close_,12)),volume_);
}

Mat Alpha191::alpha135() const {
    return Sma(Delay(Div(close_,Delay(close_,20)),1),20,1);
}

Mat Alpha191::alpha136() const {
    return Mul(ScalarMul(Rank(Delta(returns_,3)),-1.0),Corr(open_,volume_,10));
}

Mat Alpha191::alpha137() const {
    auto A=Abs(Sub(high_,Delay(close_,1)));
    auto B=Abs(Sub(low_,Delay(close_,1)));
    auto C=Abs(Sub(high_,Delay(low_,1)));
    auto D=Abs(Sub(Delay(close_,1),Delay(open_,1)));
    auto condA=mat_and(mat_gt(A,B),mat_gt(A,C));
    auto condB=mat_and(mat_gt(B,C),mat_gt(B,A));
    auto p0=ScalarMul(Sub(Add(close_,ScalarMul(Sub(close_,open_),0.5)),Delay(open_,1)),16.0);
    auto brA=Add(Add(A,ScalarMul(B,0.5)),ScalarMul(D,0.25));
    auto brB=Add(Add(B,ScalarMul(A,0.5)),ScalarMul(D,0.25));
    auto brC=Add(C,ScalarMul(D,0.25));
    auto part1=Where(condA,brA,Where(condB,brB,brC));
    return Div(Mul(p0,EwMax(A,B)),part1);
}

Mat Alpha191::alpha138() const {
    auto combo=Add(ScalarMul(low_,0.7),ScalarMul(vwap_,0.3));
    auto A=Rank(Decaylinear(Delta(combo,3),20));
    auto B=Tsrank(Decaylinear(Tsrank(Corr(Tsrank(low_,8),Tsrank(Mean(volume_,60),17),5),19),16),7);
    return ScalarMul(Sub(A,B),-1.0);
}

Mat Alpha191::alpha139() const {
    return ScalarMul(Corr(open_,volume_,10),-1.0);
}

Mat Alpha191::alpha140() const {
    auto A=Rank(Decaylinear(Sub(Add(Rank(open_),Rank(low_)),Add(Rank(high_),Rank(close_))),8));
    auto B=Tsrank(Decaylinear(Corr(Tsrank(close_,8),Tsrank(Mean(volume_,60),20),8),7),3);
    return EwMin(A,B);
}

Mat Alpha191::alpha141() const {
    return ScalarMul(Rank(Corr(Rank(high_),Rank(Mean(volume_,15)),9)),-1.0);
}

Mat Alpha191::alpha142() const {
    return Mul(Mul(ScalarMul(Rank(Tsrank(close_,10)),-1.0),
                   Rank(Delta(Delta(close_,1),1))),
               Rank(Tsrank(Div(volume_,Mean(volume_,20)),5)));
}

Mat Alpha191::alpha143() const { return zeros(); }

Mat Alpha191::alpha144() const {
    auto ret_v=Div(Abs(Sub(Div(close_,Delay(close_,1)),scalar_mat(1.0))),amount_);
    auto cond=mat_lt(close_,Delay(close_,1));
    return Div(Sumif(ret_v,20,cond),Count(cond,20));
}

Mat Alpha191::alpha145() const {
    return ScalarMul(Div(Sub(Mean(volume_,9),Mean(volume_,26)),Mean(volume_,12)),100.0);
}


/*
Mat Alpha191::alpha146() const {
    auto ret=Div(Sub(close_,Delay(close_,1)),Delay(close_,1));
    auto ema_ret=Sma(ret,61,2);
    auto dev=Sub(ret,ema_ret);
    auto mean_dev=Mean(dev,20);
    auto var_dev=Sma(Mul(Sub(ret,ret),scalar_mat(1.0)),61,2); // placeholder: should be sma(dev^2,61,2)
    // correct: sma((ret - (ret - dev))^2,61,2) = sma(dev^2,61,2)
    // Python: sma((ret - ema_ret)^2, 61, 2)
    auto sma_dev2=Sma(Mul(dev,dev),61,2);
    return Div(Mul(mean_dev,dev),sma_dev2);
}
*/


Mat Alpha191::alpha146() const {
    auto ret     = Div(Sub(close_, Delay(close_,1)), Delay(close_,1));
    auto ema_ret = Sma(ret, 61, 2);
    auto dev     = Sub(ret, ema_ret);
    auto mean_dev = Mean(dev, 20);
    // FIX: denominatore è Sma(ema_ret^2, 61, 2) non Sma(dev^2, 61, 2)
    auto sma_ema2 = Sma(Mul(ema_ret, ema_ret), 61, 2);
    return Div(Mul(mean_dev, dev), sma_ema2);
}


Mat Alpha191::alpha147() const { return Regbeta(Mean(close_,12),12); }

Mat Alpha191::alpha148() const {
    auto A=Rank(Corr(open_,Sum(Mean(volume_,60),9),6));
    auto B=Rank(Sub(open_,Tsmin(open_,14)));
    return WhereScalar(mat_lt(A,B),-1.0,0.0);
}

Mat Alpha191::alpha149() const { return zeros(); }  // requires benchmark

Mat Alpha191::alpha150() const {
    return Mul(ScalarMul(Add(Add(close_,high_),low_),1.0/3),volume_);
}

Mat Alpha191::alpha151() const {
    return Sma(Sub(close_,Delay(close_,20)),20,1);
}

Mat Alpha191::alpha152() const {
    auto inner=Sma(Delay(Div(close_,Delay(close_,9)),1),9,1);
    return Sma(Sub(Mean(Delay(inner,1),12),Mean(Delay(inner,1),26)),9,1);
}

Mat Alpha191::alpha153() const {
    return ScalarMul(Add(Add(Mean(close_,3),Mean(close_,6)),Add(Mean(close_,12),Mean(close_,24))),0.25);
}

Mat Alpha191::alpha154() const {
    auto cond=mat_lt(Sub(vwap_,Tsmin(vwap_,16)),Corr(vwap_,Mean(volume_,180),18));
    return WhereScalar(cond,1.0,0.0);
}

Mat Alpha191::alpha155() const {
    auto diff=Sub(Sma(volume_,13,2),Sma(volume_,27,2));
    return Sub(diff,Sma(diff,10,2));
}

Mat Alpha191::alpha156() const {
    auto A=Rank(Decaylinear(Delta(vwap_,5),3));
    auto combo=Add(ScalarMul(open_,0.15),ScalarMul(low_,0.85));
    auto B=Rank(Decaylinear(ScalarMul(Div(Delta(combo,2),combo),-1.0),3));
    return ScalarMul(EwMax(A,B),-1.0);
}

Mat Alpha191::alpha157() const {
    // Prod over window=1 is identity; simplify
    auto inner=ScalarMul(Rank(Delta(Sub(close_,scalar_mat(1.0)),5)),-1.0);
    auto A=Tsmin(Rank(Rank(Log(Sum(Tsmin(Rank(Rank(inner)),2),1)))),5);
    auto B=Tsrank(Delay(ScalarMul(returns_,-1.0),6),5);
    return Add(A,B);
}

Mat Alpha191::alpha158() const {
    auto sma15=Sma(close_,15,2);
    return Div(Sub(Sub(high_,sma15),Sub(low_,sma15)),close_);
}

Mat Alpha191::alpha159() const {
    auto d1=Delay(close_,1);
    auto minLD=EwMin(low_,d1);
    auto maxHD=EwMax(high_,d1);
    auto range6=Sub(maxHD,minLD), range12=range6, range24=range6;  // reuse
    auto s6 =Sum(minLD,6),  r6 =Sum(Sub(EwMax(high_,d1),EwMin(low_,d1)),6);
    auto s12=Sum(minLD,12), r12=Sum(Sub(EwMax(high_,d1),EwMin(low_,d1)),12);
    auto s24=Sum(minLD,24), r24=Sum(Sub(EwMax(high_,d1),EwMin(low_,d1)),24);
    auto t1=ScalarMul(Div(Sub(close_,s6),r6),  12.0*24);
    auto t2=ScalarMul(Div(Sub(close_,s12),r12),  6.0*24);
    auto t3=ScalarMul(Div(Sub(close_,s24),r24),  6.0*24);
    return ScalarMul(Div(Add(Add(t1,t2),t3),scalar_mat(6.0*12+6.0*24+12.0*24)),100.0);
}

Mat Alpha191::alpha160() const {
    auto cond=mat_le(close_,Delay(close_,1));
    auto std20=Std(close_,20);
    return Sma(WhereLeft(cond,std20,0.0),20,1);
}

Mat Alpha191::alpha161() const {
    auto A=Sub(high_,low_);
    auto B=Abs(Sub(Delay(close_,1),high_));
    auto C=Abs(Sub(Delay(close_,1),low_));
    return Mean(EwMax(EwMax(A,B),C),12);
}

Mat Alpha191::alpha162() const {
    auto diff=Sub(close_,Delay(close_,1));
    auto rsi=ScalarMul(Div(Sma(ScalarMax(diff,0.0),12,1),Sma(Abs(diff),12,1)),100.0);
    auto mn=Tsmin(rsi,12), mx=Tsmax(rsi,12);  // Note: Tsmax of rsi needs special path
    // We apply Tsmax/Tsmin on the rsi Mat
    return Div(Sub(rsi,mn),Sub(Sma(rsi,12,1),mn));
}

Mat Alpha191::alpha163() const {
    return Rank(Mul(Mul(Mul(ScalarMul(returns_,-1.0),Mean(volume_,20)),vwap_),Sub(high_,close_)));
}

Mat Alpha191::alpha164() const {
    int T = close_.size(), D = close_[0].size();
    
    // part: 1/diff se diff>0, else 1.0
    Mat part(T, Vec(D, NaN));
    for (int t = 0; t < T; ++t) {
        part[t][0] = NaN;
        for (int d = 1; d < D; ++d) {
            double diff = close_[t][d] - close_[t][d-1];
            part[t][d] = (diff > 0) ? (1.0 / diff) : 1.0;
        }
    }
    
    Mat tsmin_part(T, Vec(D, NaN));

    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) {

            if (d < 11) {          // window=12 → min_periods=12
                tsmin_part[t][d] = NaN;
                continue;
            }

            double mn = std::numeric_limits<double>::infinity();

            for (int k = d - 11; k <= d; ++k)
                if (!std::isnan(part[t][k]))
                    mn = std::min(mn, part[t][k]);

            tsmin_part[t][d] =
                (mn == std::numeric_limits<double>::infinity()) ? NaN : mn;
        }
    }
    
    // hl_safe: high-low con 0→NaN
    Mat hl_safe(T, Vec(D, NaN));
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d) {
            double v = high_[t][d] - low_[t][d];
            hl_safe[t][d] = (v == 0.0) ? NaN : v;
        }
    
    // inner = (part - tsmin_part) * 100 / hl_safe
    Mat inner(T, Vec(D, NaN));
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            if (!std::isnan(part[t][d]) && !std::isnan(tsmin_part[t][d]) && !std::isnan(hl_safe[t][d]))
                inner[t][d] = (part[t][d] - tsmin_part[t][d]) * 100.0 / hl_safe[t][d];
    
    return Sma(inner, 13, 2);
}


Mat Alpha191::alpha165() const { return zeros(); }  // requires rowmax/rowmin


Mat Alpha191::alpha166() const {
    auto ratio       = Div(close_, Delay(close_,1));          // close/delay
    auto ret         = Sub(ratio, scalar_mat(1.0));            // ratio - 1
    auto mean_ret    = Mean(ret, 20);                          // Mean(ret, 20)
    auto dev         = Sub(ret, mean_ret);                     // ret - mean_ret
    // p1 = -20 * 19^1.5 * Sum(dev, 20)
    auto p1          = ScalarMul(Sum(dev, 20), -20.0 * std::pow(19.0, 1.5));
    // p2 = 19*18 * Sum(Mean(ratio,20)^2, 20)^1.5
    auto mean_ratio  = Mean(ratio, 20);
    auto sum_mr2     = Sum(Mul(mean_ratio, mean_ratio), 20);
    auto p2 = ScalarMul(
        mat_apply1(sum_mr2, [](const Vec& v) {
            Vec o(v.size(), NaN);
            for (size_t i = 0; i < v.size(); ++i)
                if (!std::isnan(v[i]) && v[i] > 0) o[i] = std::pow(v[i], 1.5);
            return o;
        }),
        double(19 * 18)   // moltiplica DOPO il pow
    );
    return Div(p1, p2);
}


Mat Alpha191::alpha167() const {
    auto diff=Sub(close_,Delay(close_,1));
    auto cond=mat_gt(diff,zeros());
    return Sum(WhereLeft(cond,diff,0.0),12);
}

Mat Alpha191::alpha168() const {
    return ScalarMul(Div(volume_,Mean(volume_,20)),-1.0);
}

Mat Alpha191::alpha169() const {
    auto sma9=Sma(Sub(close_,Delay(close_,1)),9,1);
    auto inner=Delay(sma9,1);
    return Sma(Sub(Mean(inner,12),Mean(inner,26)),10,1);
}

Mat Alpha191::alpha170() const {
    auto A=Div(Mul(Div(Rank(ScalarDiv(1.0,close_)),Mean(volume_,20)),
                   Mul(high_,Rank(Sub(high_,close_)))),
               ScalarMul(Sum(high_,5),0.2));
    auto B=Rank(Sub(vwap_,Delay(vwap_,5)));
    return Sub(A,B);
}

Mat Alpha191::alpha171() const {
    auto num=ScalarMul(Mul(Sub(low_,close_),Pow(open_,scalar_mat(5.0))),-1.0);
    auto den=Mul(Sub(close_,high_),Pow(close_,scalar_mat(5.0)));
    return Div(num,den);
}

Mat Alpha191::alpha172() const {
    auto TR=EwMax(EwMax(Sub(high_,low_),Abs(Sub(high_,Delay(close_,1)))),
                  Abs(Sub(low_,Delay(close_,1))));
    auto HD=Sub(high_,Delay(high_,1));
    auto LD=Sub(Delay(low_,1),low_);
    auto condLD=mat_and(mat_gt(LD,zeros()),mat_gt(LD,HD));
    auto condHD=mat_and(mat_gt(HD,zeros()),mat_gt(HD,LD));
    auto pLD=WhereLeft(condLD,LD,0.0);
    auto pHD=WhereLeft(condHD,HD,0.0);
    auto sTR=Sum(TR,14);
    auto A=Div(ScalarMul(Sum(pLD,14),100.0),sTR);
    auto B=Div(ScalarMul(Sum(pHD,14),100.0),sTR);
    return Mean(ScalarMul(Div(Abs(Sub(A,B)),Add(A,B)),100.0),6);
}

Mat Alpha191::alpha173() const {
    return Add(Sub(ScalarMul(Sma(close_,13,2),3.0),ScalarMul(Sma(Sma(close_,13,2),13,2),2.0)),
               Sma(Sma(Sma(Log(close_),13,2),13,2),13,2));
}

Mat Alpha191::alpha174() const {
    auto cond=mat_gt(close_,Delay(close_,1));
    return Sma(WhereLeft(cond,Std(close_,20),0.0),20,1);
}

Mat Alpha191::alpha175() const {
    auto A=Sub(high_,low_);
    auto B=Abs(Sub(Delay(close_,1),high_));
    auto C=Abs(Sub(Delay(close_,1),low_));
    return Mean(EwMax(EwMax(A,B),C),6);
}

Mat Alpha191::alpha176() const {
    auto stoch=Div(Sub(close_,Tsmin(low_,12)),Sub(Tsmax(high_,12),Tsmin(low_,12)));
    return Corr(Rank(stoch),Rank(volume_),6);
}

Mat Alpha191::alpha177() const {
    return ScalarMul(Div(Sub(scalar_mat(20.0),Highday(high_,20)),scalar_mat(20.0)),100.0);
}

Mat Alpha191::alpha178() const {
    return Mul(Div(Sub(close_,Delay(close_,1)),Delay(close_,1)),volume_);
}

Mat Alpha191::alpha179() const {
    return Mul(Rank(Corr(vwap_,volume_,4)),
               Rank(Corr(Rank(low_),Rank(Mean(volume_,50)),12)));
}

/*
Mat Alpha191::alpha180() const {
    auto cond=mat_gt(Mean(volume_,20),volume_);
    auto brTrue=Mul(ScalarMul(Tsrank(Abs(Delta(close_,7)),60),-1.0),Sign(Delta(close_,7)));
    auto brFalse=ScalarMul(volume_,-1.0);
    return Where(cond,brTrue,brFalse);
}
*/

Mat Alpha191::alpha180() const {
    // FIX: cond = volume > Mean(volume,20), non Mean > volume
    auto cond    = mat_gt(volume_, Mean(volume_,20));
    auto brTrue  = Mul(ScalarMul(Tsrank(Abs(Delta(close_,7)),60),-1.0), Sign(Delta(close_,7)));
    auto brFalse = ScalarMul(volume_,-1.0);
    return Where(cond, brTrue, brFalse);
}

Mat Alpha191::alpha181() const { return zeros(); }
Mat Alpha191::alpha182() const { return zeros(); }
Mat Alpha191::alpha183() const { return zeros(); }

Mat Alpha191::alpha184() const {
    return Add(Rank(Corr(Delay(Sub(open_,close_),1),close_,200)),
               Rank(Sub(open_,close_)));
}

Mat Alpha191::alpha185() const {
    auto x=Sub(scalar_mat(1.0),Div(open_,close_));
    return Rank(ScalarMul(Mul(x,x),-1.0));
}

Mat Alpha191::alpha186() const {
    auto adx=alpha172();
    return ScalarMul(Add(adx,Delay(adx,6)),0.5);
}

Mat Alpha191::alpha187() const {
    auto cond=mat_le(open_,Delay(open_,1));
    auto mx=EwMax(Sub(high_,open_),Sub(open_,Delay(open_,1)));
    return Sum(WhereRight(cond,0.0,mx),20);
}

Mat Alpha191::alpha188() const {
    auto hl=Sub(high_,low_);
    return ScalarMul(Div(Sub(hl,Sma(hl,11,2)),Sma(hl,11,2)),100.0);
}

Mat Alpha191::alpha189() const {
    return Mean(Abs(Sub(close_,Mean(close_,6))),6);
}

Mat Alpha191::alpha190() const { return zeros(); }  // complex formula not implementable without extra data

Mat Alpha191::alpha191() const {
    return Sub(Add(Corr(Mean(volume_,20),low_,5),ScalarMul(Add(high_,low_),0.5)),close_);
}

// ===========================================================================
//  calculate() dispatch
// ===========================================================================
Mat Alpha191::calculate(int n) const {
    switch(n) {
        case 1:   return alpha001();
        case 2:   return alpha002();
        case 3:   return alpha003();
        case 4:   return alpha004();
        case 5:   return alpha005();
        case 6:   return alpha006();
        case 7:   return alpha007();
        case 8:   return alpha008();
        case 9:   return alpha009();
        case 10:  return alpha010();
        case 11:  return alpha011();
        case 12:  return alpha012();
        case 13:  return alpha013();
        case 14:  return alpha014();
        case 15:  return alpha015();
        case 16:  return alpha016();
        case 17:  return alpha017();
        case 18:  return alpha018();
        case 19:  return alpha019();
        case 20:  return alpha020();
        case 21:  return alpha021();
        case 22:  return alpha022();
        case 23:  return alpha023();
        case 24:  return alpha024();
        case 25:  return alpha025();
        case 26:  return alpha026();
        case 27:  return alpha027();
        case 28:  return alpha028();
        case 29:  return alpha029();
        case 30:  return alpha030();
        case 31:  return alpha031();
        case 32:  return alpha032();
        case 33:  return alpha033();
        case 34:  return alpha034();
        case 35:  return alpha035();
        case 36:  return alpha036();
        case 37:  return alpha037();
        case 38:  return alpha038();
        case 39:  return alpha039();
        case 40:  return alpha040();
        case 41:  return alpha041();
        case 42:  return alpha042();
        case 43:  return alpha043();
        case 44:  return alpha044();
        case 45:  return alpha045();
        case 46:  return alpha046();
        case 47:  return alpha047();
        case 48:  return alpha048();
        case 49:  return alpha049();
        case 50:  return alpha050();
        case 51:  return alpha051();
        case 52:  return alpha052();
        case 53:  return alpha053();
        case 54:  return alpha054();
        case 55:  return alpha055();
        case 56:  return alpha056();
        case 57:  return alpha057();
        case 58:  return alpha058();
        case 59:  return alpha059();
        case 60:  return alpha060();
        case 61:  return alpha061();
        case 62:  return alpha062();
        case 63:  return alpha063();
        case 64:  return alpha064();
        case 65:  return alpha065();
        case 66:  return alpha066();
        case 67:  return alpha067();
        case 68:  return alpha068();
        case 69:  return alpha069();
        case 70:  return alpha070();
        case 71:  return alpha071();
        case 72:  return alpha072();
        case 73:  return alpha073();
        case 74:  return alpha074();
        case 75:  return alpha075();
        case 76:  return alpha076();
        case 77:  return alpha077();
        case 78:  return alpha078();
        case 79:  return alpha079();
        case 80:  return alpha080();
        case 81:  return alpha081();
        case 82:  return alpha082();
        case 83:  return alpha083();
        case 84:  return alpha084();
        case 85:  return alpha085();
        case 86:  return alpha086();
        case 87:  return alpha087();
        case 88:  return alpha088();
        case 89:  return alpha089();
        case 90:  return alpha090();
        case 91:  return alpha091();
        case 92:  return alpha092();
        case 93:  return alpha093();
        case 94:  return alpha094();
        case 95:  return alpha095();
        case 96:  return alpha096();
        case 97:  return alpha097();
        case 98:  return alpha098();
        case 99:  return alpha099();
        case 100: return alpha100();
        case 101: return alpha101();
        case 102: return alpha102();
        case 103: return alpha103();
        case 104: return alpha104();
        case 105: return alpha105();
        case 106: return alpha106();
        case 107: return alpha107();
        case 108: return alpha108();
        case 109: return alpha109();
        case 110: return alpha110();
        case 111: return alpha111();
        case 112: return alpha112();
        case 113: return alpha113();
        case 114: return alpha114();
        case 115: return alpha115();
        case 116: return alpha116();
        case 117: return alpha117();
        case 118: return alpha118();
        case 119: return alpha119();
        case 120: return alpha120();
        case 121: return alpha121();
        case 122: return alpha122();
        case 123: return alpha123();
        case 124: return alpha124();
        case 125: return alpha125();
        case 126: return alpha126();
        case 127: return alpha127();
        case 128: return alpha128();
        case 129: return alpha129();
        case 130: return alpha130();
        case 131: return alpha131();
        case 132: return alpha132();
        case 133: return alpha133();
        case 134: return alpha134();
        case 135: return alpha135();
        case 136: return alpha136();
        case 137: return alpha137();
        case 138: return alpha138();
        case 139: return alpha139();
        case 140: return alpha140();
        case 141: return alpha141();
        case 142: return alpha142();
        case 143: return alpha143();
        case 144: return alpha144();
        case 145: return alpha145();
        case 146: return alpha146();
        case 147: return alpha147();
        case 148: return alpha148();
        case 149: return alpha149();
        case 150: return alpha150();
        case 151: return alpha151();
        case 152: return alpha152();
        case 153: return alpha153();
        case 154: return alpha154();
        case 155: return alpha155();
        case 156: return alpha156();
        case 157: return alpha157();
        case 158: return alpha158();
        case 159: return alpha159();
        case 160: return alpha160();
        case 161: return alpha161();
        case 162: return alpha162();
        case 163: return alpha163();
        case 164: return alpha164();
        case 165: return alpha165();
        case 166: return alpha166();
        case 167: return alpha167();
        case 168: return alpha168();
        case 169: return alpha169();
        case 170: return alpha170();
        case 171: return alpha171();
        case 172: return alpha172();
        case 173: return alpha173();
        case 174: return alpha174();
        case 175: return alpha175();
        case 176: return alpha176();
        case 177: return alpha177();
        case 178: return alpha178();
        case 179: return alpha179();
        case 180: return alpha180();
        case 181: return alpha181();
        case 182: return alpha182();
        case 183: return alpha183();
        case 184: return alpha184();
        case 185: return alpha185();
        case 186: return alpha186();
        case 187: return alpha187();
        case 188: return alpha188();
        case 189: return alpha189();
        case 190: return alpha190();
        case 191: return alpha191();
        default:  throw std::out_of_range("Alpha number must be 1-191");
    }
}

std::unordered_map<std::string, Mat> Alpha191::calculate_all() const {
    std::unordered_map<std::string, Mat> result;
    for (int i = 1; i <= 191; ++i) {
        char name[12];
        std::snprintf(name, sizeof(name), "alpha%03d", i);
        try { result[name] = calculate(i); }
        catch (const std::exception& e) {
            // fill with NaN on error
            result[name] = Mat(close_.size(), Vec(close_[0].size(), NaN));
        }
    }
    return result;
}