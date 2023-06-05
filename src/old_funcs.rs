pub fn epanechnikov_density(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    sigmaopt: f64,
) -> Array1<f64> {
    // , weights
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];

    // let points_shape = points.shape();
    // let n_points: usize = points_shape[0];
    // let n_dim_points: usize = points_shape[1];
    // should be able to check that n_dim points and n_points are the same size, so not checking over each.
    // iterate better than indexeing!

    let mut rhos = Array1::<f64>::zeros(n_stars);

    for (x_row, rho) in x.axis_iter(Axis(0)).zip(rhos.iter_mut()) {
        for (points_row, &lamda) in points.axis_iter(Axis(0)).zip(lamdaopt.iter()) {
            let t_2 = x_row
                .iter()
                .zip(points_row.iter())
                .map(|(&x_i, &points_i)| (x_i - points_i).powi(2))
                .sum::<f64>()
                / (sigmaopt * lamda).powi(2);
            if t_2 < 1. {
                *rho += (1. - t_2) / (lamda.powi(n_dim as i32));
            }
        }
    }
    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos *=
        (1. / n_stars as f64) * sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd);
    rhos
}
pub fn epanechnikov_density_par(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    sigmaopt: f64,
) -> Array1<f64> {
    println!("In par rust func!");
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];

    let mut rhos = Array1::<f64>::zeros(n_stars);
    let rhos_par = rhos.as_slice_mut().unwrap().par_iter_mut();

    // for (x_row, rho) in x.axis_iter(Axis(0)).into_par_iter().zip(rhos_par) {
    x.axis_iter(Axis(0))
        .into_par_iter()
        .zip(rhos_par)
        .for_each(|(x_row, rho)| {
            for (points_row, &lamda) in points.axis_iter(Axis(0)).zip(lamdaopt.iter()) {
                let t_2 = x_row
                    .iter()
                    .zip(points_row.iter())
                    .map(|(&x_i, &points_i)| (x_i - points_i).powi(2))
                    .sum::<f64>()
                    / (sigmaopt * lamda).powi(2);
                if t_2 < 1. {
                    *rho += (1. - t_2) / (lamda.powi(n_dim as i32));
                }
            }
        });
    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos *=
        (1. / n_stars as f64) * sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd);
    rhos
}

// impl ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>> {}

#[pyfunction]
#[pyo3(name = "epanechnikov_density")]
fn epanechnikov_density_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    sigmaopt: f64,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let res = mbe_rust_funcs::epanechnikov_density(x, points, lamdaopt, sigmaopt);
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_density_par")]
fn epanechnikov_density_par_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    sigmaopt: f64,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let res = mbe_rust_funcs::epanechnikov_density_par(x, points, lamdaopt, sigmaopt);
    res.to_pyarray(py)
}
// m.add_function(wrap_pyfunction!(epanechnikov_density_py, m)?)?;
// m.add_function(wrap_pyfunction!(epanechnikov_density_par_py, m)?)?;
