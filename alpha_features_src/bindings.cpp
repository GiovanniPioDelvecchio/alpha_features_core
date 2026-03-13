#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "alpha_features_core.h"

namespace py = pybind11;

#ifdef _WIN32
  #include <cstddef>  // per ptrdiff_t
  typedef ptrdiff_t ssize_t;
#endif

// ---------------------------------------------------------------------------
// numpy array <-> Mat conversion helpers
// ---------------------------------------------------------------------------

// numpy 2-D array (tickers × dates) → Mat
static Mat np2mat(const py::array_t<double>& arr) {
    auto r = arr.unchecked<2>();
    int T = r.shape(0), D = r.shape(1);
    Mat m(T, Vec(D));
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            m[t][d] = r(t, d);
    return m;
}

// Mat → numpy 2-D array (tickers × dates)
static py::array_t<double> mat2np(const Mat& m) {
    if (m.empty()) return py::array_t<double>(std::vector<ssize_t>{0, 0});
    int T = m.size(), D = m[0].size();
    py::array_t<double> arr(std::vector<ssize_t>{T, D});
    auto w = arr.mutable_unchecked<2>();
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            w(t, d) = m[t][d];
    return arr;
}

// ---------------------------------------------------------------------------
// Alpha191 Python wrapper
// ---------------------------------------------------------------------------
struct PyAlpha191 {
    Alpha191 impl;

    PyAlpha191(
        const py::array_t<double>& open,
        const py::array_t<double>& high,
        const py::array_t<double>& low,
        const py::array_t<double>& close,
        const py::array_t<double>& volume,
        const py::array_t<double>& amount,
        const py::array_t<double>& returns)
        : impl(np2mat(open), np2mat(high), np2mat(low), np2mat(close),
               np2mat(volume), np2mat(amount), np2mat(returns))
    {}

    py::array_t<double> calculate(int n) {
        return mat2np(impl.calculate(n));
    }

    py::dict calculate_all() {
        auto all = impl.calculate_all();
        py::dict d;
        for (auto& kv : all)
            d[py::str(kv.first)] = mat2np(kv.second);
        return d;
    }

    int n_tickers() const { return impl.n_tickers(); }
    int n_dates()   const { return impl.n_dates(); }
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(alpha_features_core, m) {
    m.doc() = "Alpha191 feature library — C++ backend with pybind11 bindings";

    // -----------------------------------------------------------------------
    // 1-D primitive functions (exposed for testing / debugging)
    // -----------------------------------------------------------------------
    m.def("ts_delta",       &ts_delta,       py::arg("x"), py::arg("period"));
    m.def("ts_delay",       &ts_delay,       py::arg("x"), py::arg("period"));
    m.def("ts_sum",         &ts_sum,         py::arg("x"), py::arg("window"));
    m.def("ts_mean",        &ts_mean,        py::arg("x"), py::arg("window"));
    m.def("ts_std",         &ts_std,         py::arg("x"), py::arg("window"));
    m.def("ts_max",         &ts_max,         py::arg("x"), py::arg("window"));
    m.def("ts_min",         &ts_min,         py::arg("x"), py::arg("window"));
    m.def("ts_rank",        &ts_rank,        py::arg("x"), py::arg("window"));
    m.def("ts_corr",        &ts_corr,        py::arg("x"), py::arg("y"), py::arg("window"));
    m.def("ts_cov",         &ts_cov,         py::arg("x"), py::arg("y"), py::arg("window"));
    m.def("ts_sma",         &ts_sma,         py::arg("x"), py::arg("n"), py::arg("m"));
    m.def("ts_wma",         &ts_wma,         py::arg("x"), py::arg("window"));
    m.def("ts_decaylinear", &ts_decaylinear, py::arg("x"), py::arg("window"));
    m.def("ts_regbeta",     &ts_regbeta,     py::arg("y"), py::arg("window"));
    m.def("ts_count",       &ts_count,       py::arg("cond"), py::arg("window"));
    m.def("ts_sumif",       &ts_sumif,       py::arg("x"), py::arg("window"), py::arg("cond"));
    m.def("ts_lowday",      &ts_lowday,      py::arg("x"), py::arg("window"));
    m.def("ts_highday",     &ts_highday,     py::arg("x"), py::arg("window"));
    m.def("ew_log",         &ew_log,         py::arg("x"));
    m.def("ew_sign",        &ew_sign,        py::arg("x"));
    m.def("ew_abs",         &ew_abs,         py::arg("x"));

    // -----------------------------------------------------------------------
    // Alpha191 class
    // -----------------------------------------------------------------------
    py::class_<PyAlpha191>(m, "Alpha191",
        R"doc(
        Alpha191 factor library.

        Parameters
        ----------
        open, high, low, close, volume, amount, returns : np.ndarray, shape (n_tickers, n_dates)
            All arrays must have the same shape.
            ``amount`` = close * volume (turnover).
            ``returns`` = pct_change of close (1-period).

        Layout
        ------
        Column-major: axis-0 = tickers, axis-1 = dates (time), sorted ascending.

        Usage
        -----
        >>> calc = alpha_features_core.Alpha191(open, high, low, close, volume, amount, returns)
        >>> a1  = calc.calculate(1)          # shape (n_tickers, n_dates)
        >>> all = calc.calculate_all()       # dict[str, np.ndarray]
        )doc")
        .def(py::init<
                const py::array_t<double>&,
                const py::array_t<double>&,
                const py::array_t<double>&,
                const py::array_t<double>&,
                const py::array_t<double>&,
                const py::array_t<double>&,
                const py::array_t<double>&>(),
             py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"),
             py::arg("volume"), py::arg("amount"), py::arg("returns"))
        .def("calculate", &PyAlpha191::calculate,
             py::arg("alpha_num"),
             "Compute a single alpha (1-191). Returns np.ndarray of shape (n_tickers, n_dates).")
        .def("calculate_all", &PyAlpha191::calculate_all,
             "Compute all 191 alphas. Returns dict[str, np.ndarray].")
        .def_property_readonly("n_tickers", &PyAlpha191::n_tickers)
        .def_property_readonly("n_dates",   &PyAlpha191::n_dates);
}