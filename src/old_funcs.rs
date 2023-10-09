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

pub fn epanechnikov_density_kde_3d_rev_weights(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    let points_shape = points.shape();
    let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];

    assert_eq!(n_dim, n_dim_points); //else ndarray to array3 is not allowed!

    let mut rhos = Array1::<f64>::zeros(n_stars);
    let lamdaopt_sigma2: Array1<f64> = lamdaopt.map(|&x| x * x * sigmaopt * sigmaopt);
    let w_inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(n_dim as i32))) * weights;

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_stars / n_threads, 50_000), 10_000);
    // println!("New parallel reverse fixed : {}", n_chunk);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_small)| {
                let mut stars_kdtree: KdTree<f64, n_dim_points> = KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt_sigma2)
                    .and(&w_inv_lamdaopt_pow)
                    .for_each(|p_row, lamda_s2, w_inv_lamda| {
                        let neighbours = stars_kdtree.within_unsorted(
                            ndarray_to_array3(p_row.to_slice().unwrap()),
                            *lamda_s2,
                            &squared_euclidean,
                        );
                        for neigh in neighbours {
                            let t_2 = neigh.distance / lamda_s2;
                            rhos_small[neigh.item] += (1. - t_2) * w_inv_lamda;
                        }
                    });
            });
    });

    //
    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos *=
        (1. / n_points as f64) * sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd); //can
                                                                                                   //include the sigmaopt earlier
    rhos
}



