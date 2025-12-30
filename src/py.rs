#![allow(deprecated)]
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::types::PyAny;
use pyo3::exceptions::PyTypeError;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArrayDyn};

/// Python-visible mirror of `crate::harvest::HarvestOption`.
#[pyclass(name = "HarvestOption")]
#[derive(Clone, Debug)]
pub struct PyHarvestOption {
    #[pyo3(get, set)]
    pub f0_floor: f64,
    #[pyo3(get, set)]
    pub f0_ceil: f64,
    /// Unit: ms
    #[pyo3(get, set)]
    pub frame_period: f64,
}

#[pymethods]
impl PyHarvestOption {
    #[new]
    #[pyo3(signature = (f0_floor = 71.0, f0_ceil = 800.0, frame_period = 5.0))]
    fn new(f0_floor: f64, f0_ceil: f64, frame_period: f64) -> Self {
        Self {
            f0_floor,
            f0_ceil,
            frame_period,
        }
    }
}

impl PyHarvestOption {
    fn to_rs(&self) -> crate::harvest::HarvestOption {
        crate::harvest::HarvestOption {
            f0_floor: self.f0_floor,
            f0_ceil: self.f0_ceil,
            frame_period: self.frame_period,
        }
    }
}


/// Input: `x` is a float64 numpy array with shape [B, SEQ].
/// Output: (temporal_positions, f0), both float64 numpy arrays with shape [B, N].
#[pyfunction]
#[pyo3(signature = (x, fs, option=None))]
pub fn harvest(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    fs: i32,
    option: Option<&PyHarvestOption>,
) -> PyResult<(PyObject, PyObject)> {
    let opt = option.map(|o| o.to_rs()).unwrap_or_default();

    // Prefer float64, but accept float32 (e.g., librosa default).
    if let Ok(x64) = x.extract::<PyReadonlyArrayDyn<f64>>() {
        return harvest_impl_f64(py, x64, fs, &opt);
    }
    if let Ok(x32) = x.extract::<PyReadonlyArrayDyn<f32>>() {
        // Convert once to f64, then reuse the same core implementation.
        let x_view32 = x32.as_array();
        let x_owned32: Option<numpy::ndarray::ArrayD<f32>> = if x_view32.is_standard_layout() {
            None
        } else {
            Some(x_view32.to_owned())
        };
        let x_flat32: &[f32] = match &x_owned32 {
            Some(a) => a.as_slice().ok_or_else(|| PyValueError::new_err(
                "Cannot convert x to continuous memory (standard layout), Please use numpy.ascontiguousarray to convert x to continuous memory"
            ))?,
            None => x_view32
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("Cannot get continuous slice of x"))?,
        };

        let x_flat64: Vec<f64> = x_flat32.iter().map(|&v| v as f64).collect();
        let x64_owned = numpy::ndarray::ArrayD::<f64>::from_shape_vec(x_view32.raw_dim(), x_flat64)
            .map_err(|e| PyValueError::new_err(format!("failed to build f64 array: {e}")))?;
        // Safety: owned ndarray is contiguous standard layout by construction.
        let x64_py = x64_owned.into_pyarray(py);
        let x64 = x64_py.readonly();
        return harvest_impl_f64(py, x64, fs, &opt);
    }

    Err(PyTypeError::new_err(
        "x 必须是 numpy.ndarray，且 dtype 为 float32 或 float64",
    ))
}

fn harvest_impl_f64(
    py: Python<'_>,
    x: PyReadonlyArrayDyn<f64>,
    fs: i32,
    opt: &crate::harvest::HarvestOption,
) -> PyResult<(PyObject, PyObject)> {
    let x_view = x.as_array();
    let dim_num = x_view.ndim();
    if !(dim_num == 1 || dim_num == 2) {
        return Err(PyValueError::new_err(format!(
            "x must be a 1D or 2D array, but got {dim_num}D array, Please convert x to 1D or 2D array"
        )));
    }

    // 1D: x is [SEQ] -> return [N]
    if dim_num == 1 {
        let seq = x_view.len();
        if seq == 0 {
            return Err(PyValueError::new_err("Sequence length cannot be 0"));
        }

        let x_owned: Option<numpy::ndarray::ArrayD<f64>> = if x_view.is_standard_layout() {
            None
        } else {
            Some(x_view.to_owned())
        };
        let x_flat: &[f64] = match &x_owned {
            Some(a) => a.as_slice().ok_or_else(|| PyValueError::new_err(
                "Cannot convert x to continuous memory (standard layout), Please use numpy.ascontiguousarray to convert x to continuous memory"
            ))?,
            None => x_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("Cannot get continuous slice of x"))?,
        };

        let (t0, f0_0) = py.allow_threads(|| crate::harvest_fast_2::harvest(x_flat, fs, opt));
        let t_py: Py<PyArray1<f64>> = t0.into_pyarray(py).into();
        let f0_py: Py<PyArray1<f64>> = f0_0.into_pyarray(py).into();
        return Ok((t_py.into_any(), f0_py.into_any()));
    }

    // 2D: x is [B, SEQ] -> return [B, N]
    let shape = x_view.shape();
    let (b, seq) = (shape[0], shape[1]);
    if seq == 0 {
        return Err(PyValueError::new_err("Sequence length cannot be 0"));
    }
    if b == 0 {
        let empty_t: Py<PyArray2<f64>> = Array2::<f64>::zeros((0, 0)).into_pyarray(py).into();
        let empty_f0: Py<PyArray2<f64>> = Array2::<f64>::zeros((0, 0)).into_pyarray(py).into();
        return Ok((empty_t.into_any(), empty_f0.into_any()));
    }

    let x_owned: Option<numpy::ndarray::ArrayD<f64>> = if x_view.is_standard_layout() {
        None
    } else {
        Some(x_view.to_owned())
    };
    let x_flat: &[f64] = match &x_owned {
        Some(a) => a.as_slice().ok_or_else(|| PyValueError::new_err(
            "Cannot convert x to continuous memory (standard layout), Please use numpy.ascontiguousarray to convert x to continuous memory"
        ))?,
        None => x_view
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("Cannot get continuous slice of x"))?,
    };

    let (out_t, out_f0) = py.allow_threads(|| {
        // First row determines N and temporal_positions.
        let x0 = &x_flat[0..seq];
        let (t0, f0_0) = crate::harvest_fast_2::harvest(x0, fs, opt);
        let n = f0_0.len();

        let mut t_out = Array2::<f64>::zeros((b, n));
        let mut f0_out = Array2::<f64>::zeros((b, n));

        // Fill row 0.
        for j in 0..n {
            t_out[(0, j)] = t0[j];
            f0_out[(0, j)] = f0_0[j];
        }

        // Remaining rows.
        for i in 1..b {
            let start = i * seq;
            let end = start + seq;
            let xi = &x_flat[start..end];
            let (_ti, f0_i) = crate::harvest_fast_2::harvest(xi, fs, opt);
            if f0_i.len() != n {
                return Err(PyValueError::new_err(format!(
                    "Output length mismatch: row0 N={n}, row{i} N={}",
                    f0_i.len()
                )));
            }
            for j in 0..n {
                t_out[(i, j)] = t0[j];
                f0_out[(i, j)] = f0_i[j];
            }
        }

        Ok::<_, PyErr>((t_out, f0_out))
    })?;

    let t_py: Py<PyArray2<f64>> = out_t.into_pyarray(py).into();
    let f0_py: Py<PyArray2<f64>> = out_f0.into_pyarray(py).into();
    Ok((t_py.into_any(), f0_py.into_any()))
}

#[pymodule]
fn harvest_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHarvestOption>()?;
    m.add_function(wrap_pyfunction!(harvest, m)?)?;
    Ok(())
}


