use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
mod mbe_rust_funcs;

#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_2d")]
fn epanechnikov_density_kde_2d_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let res = mbe_rust_funcs::epanechnikov_density_kde_2d(x, points, lamdaopt, sigmaopt, n_threads);
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_3d")]
fn epanechnikov_density_kde_3d_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();

    let res = mbe_rust_funcs::epanechnikov_density_kde_3d(x, points, lamdaopt, sigmaopt, n_threads);
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_3d_weights")]
fn epanechnikov_density_kde_3d_weights_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let weights = weights.as_array();

    let res = mbe_rust_funcs::epanechnikov_density_kde_3d_weights(
        x, points, lamdaopt, sigmaopt, n_threads, weights,
    );
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_3d_rev_weights")]
fn epanechnikov_density_kde_3d_rev_weights_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    println!("In rust translate, using weights!");
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let weights = weights.as_array();
    let res = mbe_rust_funcs::epanechnikov_density_kde_3d_rev_weights(
        x, points, lamdaopt, weights, sigmaopt, n_threads,
    );
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_3d_rev")]
fn epanechnikov_density_kde_3d_rev_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let res =
        mbe_rust_funcs::epanechnikov_density_kde_3d_rev(x, points, lamdaopt, sigmaopt, n_threads);
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_2d_rev")]
fn epanechnikov_density_kde_2d_rev_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let res =
        mbe_rust_funcs::epanechnikov_density_kde_2d_rev(x, points, lamdaopt, sigmaopt, n_threads);
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_2d_weights")]
fn epanechnikov_density_kde_2d_weights_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let weights = weights.as_array();

    let res = mbe_rust_funcs::epanechnikov_density_kde_2d_weights(
        x, points, lamdaopt, sigmaopt, n_threads, weights,
    );
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_2d_rev_weights")]
fn epanechnikov_density_kde_2d_rev_weights_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    println!("In rust translate, using weights!");
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let weights = weights.as_array();
    let res = mbe_rust_funcs::epanechnikov_density_kde_2d_rev_weights(
        x, points, lamdaopt, weights, sigmaopt, n_threads,
    );
    res.to_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn mbe_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(epanechnikov_density_kde_3d_py, m)?)?;
    m.add_function(wrap_pyfunction!(epanechnikov_density_kde_3d_rev_py, m)?)?;
    m.add_function(wrap_pyfunction!(epanechnikov_density_kde_3d_weights_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        epanechnikov_density_kde_3d_rev_weights_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(epanechnikov_density_kde_2d_py, m)?)?;
    m.add_function(wrap_pyfunction!(epanechnikov_density_kde_2d_rev_py, m)?)?;
    m.add_function(wrap_pyfunction!(epanechnikov_density_kde_2d_weights_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        epanechnikov_density_kde_2d_rev_weights_py,
        m
    )?)?;
    Ok(())
}
