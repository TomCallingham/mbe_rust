use itertools::izip;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use ndarray_linalg::{Determinant, Inverse};
use spfunc::gamma;
use std::f64::consts::PI;
/* use kiddo::distance::squared_euclidean;
use kiddo::float::kdtree::KdTree as float_KdTree;
use kiddo::KdTree;
use ndarray_linalg::{solve::InverseInto, Determinant, Inverse};
use ndarray_linalg::{Eigh, UPLO};
use ndarray_stats::{self, QuantileExt}; */

fn create_pool(n_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap()
}

pub fn multi_mbe_shaped_3d_within(
    x: ArrayView2<f64>,
    multi_points: Vec<ArrayView2<f64>>,
    multi_covars: Vec<ArrayView2<f64>>,
    multi_weights: Vec<ArrayView1<f64>>,
    multi_within_ind: Vec<ArrayView1<usize>>,
    n_threads: usize,
) -> Array2<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!

    let n_groups = multi_points.len();
    let mut rhos_2d = Array2::<f64>::zeros((n_stars, n_groups));
    // create_pool(n_threads).install(|| {
    for (mut g_rhos, points, weights, covar, within_ind) in izip!(
        rhos_2d.axis_iter_mut(Axis(1)),
        multi_points.iter(),
        multi_weights.iter(),
        multi_covars.iter(),
        multi_within_ind.iter()
    ) {
        let points_shape = points.shape();
        let n_dim_points: usize = points_shape[1];
        assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!
        let cov_inv = covar.inv().expect("Expect an Inversable covar");
        let cov_det = covar.det().expect("Expect a Valid Determinant for covar");
        // for ind in within_ind.axis_iter(Axis(0)).into_par_iter() {
        for ind in within_ind.iter() {
            // for ind in within_ind.into_par_iter() {
            let x_row = x.row(*ind);
            let mut rho = 0.;
            for (x_point, w) in izip!(points.axis_iter(Axis(0)), weights) {
                let t_2 = mahalanobis_distance2(&x_row, &x_point, &cov_inv);
                if t_2 < 1. {
                    rho += w * (1. - t_2)
                }
            }
            g_rhos[*ind] = rho
        }
        g_rhos /= f64::sqrt(cov_det);
    }
    // });

    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos_2d *= (n_dim as f64 + 2.) / (2. * vd);
    rhos_2d
}

pub fn several_mahalanobis_distance2(
    x_array: &ArrayView2<f64>,
    mean: &ArrayView1<f64>,
    covar: &ArrayView2<f64>,
) -> Array1<f64> {
    let cov_inv = covar.inv().unwrap();
    let n_stars = x_array.shape()[0];
    let mut maha_d2_array = Array1::<f64>::zeros(n_stars);
    Zip::from(x_array.axis_iter(Axis(0)))
        .and(&mut maha_d2_array)
        .into_par_iter()
        .for_each(|(x_row, maha_d2)| {
            *maha_d2 = mahalanobis_distance2(&x_row, mean, &cov_inv);
        });

    maha_d2_array
}

fn mahalanobis_distance2(
    x: &ArrayView1<f64>,
    mean: &ArrayView1<f64>,
    covariance_inv: &Array2<f64>,
) -> f64 {
    let x_shift = x - mean;
    x_shift.dot(&covariance_inv.dot(&x_shift))
}
