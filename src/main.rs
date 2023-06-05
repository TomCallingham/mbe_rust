mod mbe_rust_funcs;
use ndarray::Array;
use ndarray_rand::{rand_distr::Normal, RandomExt};

fn main() {
    let n_points: usize = 1e5 as usize;
    let n_x: usize = 1e6 as usize;

    let test_points = Array::random((n_points, 3), Normal::new(0., 1.).unwrap());

    let test_x = Array::random((n_x, 3), Normal::new(0., 1.).unwrap());

    let n_threads = 10;
    let lamdaopt = Array::random(n_points, Normal::new(0., 0.2).unwrap());
    let sigmaopt = 0.5;

    let result = mbe_rust_funcs::epanechnikov_density_kde_3d(
        test_x.view(),
        test_points.view(),
        lamdaopt.view(),
        sigmaopt,
        n_threads,
    );
    let mean = result.mean().unwrap();
    println!("mean:{}", mean)
}
