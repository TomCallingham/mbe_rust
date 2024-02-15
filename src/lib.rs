use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyList;
mod mbe_rust_funcs;
mod mbe_shaped_funcs;

#[pyfunction]
#[pyo3(name = "multi_within_kde_3d")]
fn multi_within_kde_3d_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    multi_points: Vec<PyReadonlyArray2<f64>>,
    multi_lamdaopt: Vec<f64>,
    n_threads: usize,
) -> &'py PyArray2<bool> {
    let x = x.as_array();

    let vec_multi_points = multi_points.iter().map(|item| item.as_array()).collect();

    let res = mbe_rust_funcs::multi_within_kde_3d(x, vec_multi_points, multi_lamdaopt, n_threads);
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "several_mahalanobis_distance2")]
fn several_mahalanobis_distance2_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    mean: PyReadonlyArray1<f64>,
    covar: PyReadonlyArray2<f64>,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let mean = mean.as_array();
    let covar = covar.as_array();

    let maha_dis2 = mbe_shaped_funcs::several_mahalanobis_distance2(&x, &mean, &covar);

    maha_dis2.to_pyarray(py)
}

// pub fn multi_mbe_shaped_3d(
#[pyfunction]
#[pyo3(name = "multi_mbe_shaped_3d_within")]
fn multi_mbe_shaped_3d_within_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    multi_points: Vec<PyReadonlyArray2<f64>>,
    multi_covars: Vec<PyReadonlyArray2<f64>>,
    multi_weights: Vec<PyReadonlyArray1<f64>>,
    multi_within_ind: Vec<PyReadonlyArray1<usize>>,
    n_threads: usize,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    let vec_multi_points = multi_points.iter().map(|item| item.as_array()).collect();
    let vec_multi_covars = multi_covars.iter().map(|item| item.as_array()).collect();
    let vec_multi_weights = multi_weights.iter().map(|item| item.as_array()).collect();
    let vec_multi_within_ind = multi_within_ind
        .iter()
        .map(|item| item.as_array())
        .collect();

    let res = mbe_shaped_funcs::multi_mbe_shaped_3d_within(
        x,
        vec_multi_points,
        vec_multi_covars,
        vec_multi_weights,
        vec_multi_within_ind,
        n_threads,
    );
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
#[pyo3(name = "epanechnikov_density_kde_3d_rev_weights_groups")]
fn epanechnikov_density_kde_3d_rev_weights_groups_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    group_inds: PyReadonlyArray1<usize>,
    n_groups: usize,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let weights = weights.as_array();
    let group_inds = group_inds.as_array();

    let res = mbe_rust_funcs::epanechnikov_density_kde_3d_rev_weights_groups(
        x, points, lamdaopt, weights, group_inds, n_groups, sigmaopt, n_threads,
    );
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_2d_rev_weights_groups")]
fn epanechnikov_density_kde_2d_rev_weights_groups_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    group_inds: PyReadonlyArray1<usize>,
    n_groups: usize,
    sigmaopt: f64,
    n_threads: usize,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let weights = weights.as_array();
    let group_inds = group_inds.as_array();

    let res = mbe_rust_funcs::epanechnikov_density_kde_2d_rev_weights_groups(
        x, points, lamdaopt, weights, group_inds, n_groups, sigmaopt, n_threads,
    );
    res.to_pyarray(py)
}
#[pyfunction]
#[pyo3(name = "epanechnikov_density_kde_3d_rev_weights_multi")]
fn epanechnikov_density_kde_3d_rev_weights_multi_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    // multi_points: Vec<PyReadonlyArray2<'py, f64>>,
    multi_points: Vec<PyReadonlyArray2<f64>>,
    multi_lamdaopt: Vec<PyReadonlyArray1<f64>>,
    multi_weights: Vec<PyReadonlyArray1<f64>>,
    multi_sigmaopt: &'py PyList, //<f64>,
    n_threads: usize,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    println!("In rust trans multi");

    let vec_multi_points = multi_points.iter().map(|item| item.as_array()).collect();
    let vec_multi_weights = multi_weights.iter().map(|item| item.as_array()).collect();
    let vec_multi_lamdaopt = multi_lamdaopt.iter().map(|item| item.as_array()).collect();

    let multi_sigmaopt = multi_sigmaopt.extract().unwrap();

    let res = mbe_rust_funcs::epanechnikov_density_kde_3d_rev_weights_multi(
        x,
        vec_multi_points,
        vec_multi_lamdaopt,
        vec_multi_weights,
        multi_sigmaopt,
        n_threads,
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
    m.add_function(wrap_pyfunction!(
        epanechnikov_density_kde_3d_rev_weights_multi_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        epanechnikov_density_kde_3d_rev_weights_groups_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        epanechnikov_density_kde_2d_rev_weights_groups_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multi_within_kde_3d_py, m)?)?;
    m.add_function(wrap_pyfunction!(multi_mbe_shaped_3d_within_py, m)?)?;
    m.add_function(wrap_pyfunction!(several_mahalanobis_distance2_py, m)?)?;
    Ok(())
}
