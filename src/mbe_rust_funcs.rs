extern crate blas_src;
use kiddo::distance::squared_euclidean;
use kiddo::KdTree;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis, Zip};
use ndarray_stats::{self, QuantileExt};
use spfunc::gamma;
use std::f64::consts::PI;

fn create_pool(n_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap()
}

fn ndarray_to_array3(slice: &[f64]) -> &[f64; 3] {
    let array_ref: &[f64; 3] = unsafe { &*(slice.as_ptr() as *const [f64; 3]) };
    array_ref
}

pub fn epanechnikov_density_kde_3d(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];

    let points_shape = points.shape();
    let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];

    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!
                                 //
    let mut rhos = Array1::<f64>::zeros(n_stars);

    let mut kdtree: KdTree<f64, 3> = KdTree::with_capacity(n_points);
    for (idx, jvec) in points.axis_iter(Axis(0)).enumerate() {
        kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
    }

    // Could chunk to seperate max distts!
    let max_dist2: f64 = (lamdaopt.max().unwrap() * sigmaopt).powi(2);

    create_pool(n_threads).install(|| {
        Zip::from(x.axis_iter(Axis(0)))
            .and(&mut rhos)
            .into_par_iter()
            .for_each(|(x_row, rho)| {
                let neighbours = kdtree.within_unsorted(
                    ndarray_to_array3(x_row.to_slice().unwrap()),
                    max_dist2,
                    &squared_euclidean,
                );
                for neigh in neighbours {
                    let lamda = lamdaopt[neigh.item];
                    let t_2 = neigh.distance / (sigmaopt * lamda).powi(2);
                    if t_2 < 1. {
                        *rho += (1. - t_2) / (lamda.powi(n_dim as i32));
                    }
                }
            });
    });

    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    let constant_factor =
        (1. / n_points as f64) * sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd);
    rhos *= constant_factor;
    rhos
}

pub fn epanechnikov_density_kde_3d_weights(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    sigmaopt: f64,
    n_threads: usize,
    weights: ArrayView1<f64>,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];

    let points_shape = points.shape();
    let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];

    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!
                                 //
    let mut rhos = Array1::<f64>::zeros(n_stars);

    let mut kdtree: KdTree<f64, 3> = KdTree::with_capacity(n_points);
    for (idx, jvec) in points.axis_iter(Axis(0)).enumerate() {
        kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
    }

    // Could chunk to seperate max distts!
    let max_dist2: f64 = (lamdaopt.max().unwrap() * sigmaopt).powi(2);

    create_pool(n_threads).install(|| {
        Zip::from(x.axis_iter(Axis(0)))
            .and(&mut rhos)
            .into_par_iter()
            .for_each(|(x_row, rho)| {
                let neighbours = kdtree.within_unsorted(
                    ndarray_to_array3(x_row.to_slice().unwrap()),
                    max_dist2,
                    &squared_euclidean,
                );
                for neigh in neighbours {
                    let lamda = unsafe { *lamdaopt.uget(neigh.item) };
                    let w = unsafe { *weights.uget(neigh.item) };
                    let t_2 = neigh.distance / (sigmaopt * lamda).powi(2);
                    if t_2 < 1. {
                        *rho += w * (1. - t_2) / (lamda.powi(n_dim as i32));
                    }
                }
            });
    });

    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    let constant_factor =
        (1. / n_points as f64) * sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd);
    rhos *= constant_factor;
    rhos
}

pub fn epanechnikov_density_kde_3d_rev(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    let points_shape = points.shape();
    let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!

    let mut rhos = Array1::<f64>::zeros(n_stars);
    let lamdaopt_sigma2: Array1<f64> = lamdaopt.map(|&x| x * x * sigmaopt * sigmaopt);
    let inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(n_dim as i32)));

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_stars / n_threads, 50_000), 10_000);
    // println!("New parallel reverse fixed : {}", n_chunk);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_small)| {
                let mut stars_kdtree: KdTree<f64, 3> = KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt_sigma2)
                    .and(&inv_lamdaopt_pow)
                    .for_each(|p_row, lamda_s2, inv_lamda| {
                        let neighbours = stars_kdtree.within_unsorted(
                            ndarray_to_array3(p_row.to_slice().unwrap()),
                            *lamda_s2,
                            &squared_euclidean,
                        );
                        for neigh in neighbours {
                            let t_2 = neigh.distance / lamda_s2;
                            rhos_small[neigh.item] += (1. - t_2) * inv_lamda;
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
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!

    let mut rhos = Array1::<f64>::zeros(n_stars);
    let lamdaopt_sigma2: Array1<f64> = lamdaopt.map(|&x| x * x * sigmaopt * sigmaopt);
    let w_inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(n_dim as i32))) * weights;

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_stars / n_threads, 50_000), 10_000);
    // println!("New parallel reverse fixed : {}", n_chunk);

    println!("Using Weights!");
    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_small)| {
                let mut stars_kdtree: KdTree<f64, 3> = KdTree::with_capacity(n_chunk);
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

fn ndarray_to_array2(slice: &[f64]) -> &[f64; 2] {
    let array_ref: &[f64; 2] = unsafe { &*(slice.as_ptr() as *const [f64; 2]) };
    array_ref
}

pub fn epanechnikov_density_kde_2d(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    let points_shape = points.shape();
    let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];
    assert_eq!(n_dim, 2); //else ndarray to array2 is not allowed!
    assert_eq!(n_dim_points, 2); //else ndarray to array2 is not allowed!
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    let mut rhos = Array1::<f64>::zeros(n_stars);
    let rhos_par = rhos.as_slice_mut().unwrap().par_iter_mut();

    let mut kdtree: KdTree<f64, 2> = KdTree::with_capacity(n_stars);
    for (idx, jvec) in points.axis_iter(Axis(0)).enumerate() {
        kdtree.add(ndarray_to_array2(jvec.to_slice().unwrap()), idx)
    }

    let max_dist2: f64 = (lamdaopt
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap()
        * sigmaopt)
        .powi(2);

    x.axis_iter(Axis(0))
        .into_par_iter() //.into_par_iter()
        .zip(rhos_par) //.zip(rhos.iter_mut())
        .for_each(|(x_row, rho)| {
            let neighbours = kdtree.within_unsorted(
                ndarray_to_array2(x_row.to_slice().unwrap()),
                max_dist2,
                &squared_euclidean,
            );
            for neigh in neighbours {
                let lamda = lamdaopt[neigh.item];
                let t_2 = neigh.distance / (sigmaopt * lamda).powi(2);
                if t_2 < 1. {
                    *rho += (1. - t_2) / (lamda.powi(n_dim as i32));
                }
            }
        });

    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos *=
        (1. / n_points as f64) * sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd);
    rhos
}

pub fn epanechnikov_density_kde_2d_rev(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    sigmaopt: f64,
    n_threads: usize,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    let points_shape = points.shape();
    let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];
    assert_eq!(n_dim, 2); //else ndarray to array2 is not allowed!
    assert_eq!(n_dim_points, 2); //else ndarray to array2 is not allowed!

    let mut rhos = Array1::<f64>::zeros(n_stars);
    let lamdaopt_sigma2: Array1<f64> = lamdaopt.map(|&x| x * x * sigmaopt * sigmaopt);
    let inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(n_dim as i32)));

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_stars / n_threads, 50_000), 10_000);
    // println!("New parallel reverse fixed : {}", n_chunk);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_small)| {
                let mut stars_kdtree: KdTree<f64, 2> = KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(ndarray_to_array2(jvec.to_slice().unwrap()), idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt_sigma2)
                    .and(&inv_lamdaopt_pow)
                    .for_each(|p_row, lamda_s2, inv_lamda| {
                        let neighbours = stars_kdtree.within_unsorted(
                            ndarray_to_array2(p_row.to_slice().unwrap()),
                            *lamda_s2,
                            &squared_euclidean,
                        );
                        for neigh in neighbours {
                            let t_2 = neigh.distance / lamda_s2;
                            rhos_small[neigh.item] += (1. - t_2) * inv_lamda;
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
pub fn epanechnikov_density_kde_2d_rev_weights(
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
    assert_eq!(n_dim, 2); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 2); //else ndarray to array3 is not allowed!

    let mut rhos = Array1::<f64>::zeros(n_stars);
    let lamdaopt_sigma2: Array1<f64> = lamdaopt.map(|&x| x * x * sigmaopt * sigmaopt);
    let w_inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(n_dim as i32))) * weights;

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_stars / n_threads, 50_000), 10_000);
    // println!("New parallel reverse fixed : {}", n_chunk);

    println!("Using Weights!");
    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_small)| {
                let mut stars_kdtree: KdTree<f64, 2> = KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(ndarray_to_array2(jvec.to_slice().unwrap()), idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt_sigma2)
                    .and(&w_inv_lamdaopt_pow)
                    .for_each(|p_row, lamda_s2, w_inv_lamda| {
                        let neighbours = stars_kdtree.within_unsorted(
                            ndarray_to_array2(p_row.to_slice().unwrap()),
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

pub fn epanechnikov_density_kde_2d_weights(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    sigmaopt: f64,
    n_threads: usize,
    weights: ArrayView1<f64>,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];

    let points_shape = points.shape();
    let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];

    assert_eq!(n_dim, 2); //else ndarray to array2 is not allowed!
    assert_eq!(n_dim_points, 2); //else ndarray to array2 is not allowed!
                                 //
    let mut rhos = Array1::<f64>::zeros(n_stars);

    let mut kdtree: KdTree<f64, 2> = KdTree::with_capacity(n_points);
    for (idx, jvec) in points.axis_iter(Axis(0)).enumerate() {
        kdtree.add(ndarray_to_array2(jvec.to_slice().unwrap()), idx)
    }

    // Could chunk to seperate max distts!
    let max_dist2: f64 = (lamdaopt.max().unwrap() * sigmaopt).powi(2);

    create_pool(n_threads).install(|| {
        Zip::from(x.axis_iter(Axis(0)))
            .and(&mut rhos)
            .into_par_iter()
            .for_each(|(x_row, rho)| {
                let neighbours = kdtree.within_unsorted(
                    ndarray_to_array2(x_row.to_slice().unwrap()),
                    max_dist2,
                    &squared_euclidean,
                );
                for neigh in neighbours {
                    let lamda = unsafe { *lamdaopt.uget(neigh.item) };
                    let w = unsafe { *weights.uget(neigh.item) };
                    let t_2 = neigh.distance / (sigmaopt * lamda).powi(2);
                    if t_2 < 1. {
                        *rho += w * (1. - t_2) / (lamda.powi(n_dim as i32));
                    }
                }
            });
    });

    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    let constant_factor =
        (1. / n_points as f64) * sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd);
    rhos *= constant_factor;
    rhos
}
