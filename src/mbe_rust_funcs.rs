use itertools::izip;
use kiddo::distance::squared_euclidean;
use kiddo::float::kdtree::KdTree as float_KdTree;
use kiddo::KdTree;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
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
fn ndarray_to_array2(slice: &[f64]) -> &[f64; 2] {
    let array_ref: &[f64; 2] = unsafe { &*(slice.as_ptr() as *const [f64; 2]) };
    array_ref
}

pub fn multi_within_kde_3d(
    x: ArrayView2<f64>,
    multi_points: Vec<ArrayView2<f64>>,
    multi_lamdaopt: Vec<f64>,
    n_threads: usize,
) -> Array2<bool> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!

    //
    let n_groups = multi_points.len();
    let mut within_2d = Array2::from_elem((n_stars, n_groups), false);

    for (mut within, points, lamdaopt) in izip!(
        within_2d.axis_iter_mut(Axis(1)),
        multi_points.iter(),
        multi_lamdaopt.iter()
    ) {
        let max_dist2: f64 = lamdaopt.powi(2);
        let points_shape = points.shape();
        let n_points: usize = points_shape[0];
        let n_dim_points: usize = points_shape[1];
        assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!
        let mut kdtree: KdTree<f64, 3> = KdTree::with_capacity(n_points);
        for (idx, jvec) in points.axis_iter(Axis(0)).enumerate() {
            kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
        }

        create_pool(n_threads).install(|| {
            Zip::from(x.axis_iter(Axis(0)))
                .and(&mut within)
                .into_par_iter()
                .for_each(|(x_row, point_within)| {
                    let neighbours = kdtree.within_unsorted(
                        ndarray_to_array3(x_row.to_slice().unwrap()),
                        max_dist2,
                        &squared_euclidean,
                    );
                    *point_within = neighbours.len() > 0;
                });
        });
    }

    within_2d
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
    let constant_factor = sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd);
    // *(1. / n_points as f64)
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
    let constant_factor = sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd);
    // (1. / n_points as f64)
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
    // let n_points: usize = points_shape[0];
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
    rhos *= sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd); //can
                                                                              // *(1. / n_points as f64)
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
    // let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!

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
    rhos *= sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd); //can
                                                                              // *(1. / n_points as f64)
                                                                              //include the sigmaopt earlier
    rhos
}

pub fn epanechnikov_density_kde_3d_rev_weights_groups(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    group_inds: ArrayView1<usize>,
    n_groups: usize,
    sigmaopt: f64,
    n_threads: usize,
) -> Array2<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    let points_shape = points.shape();
    // let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!

    let mut rhos_2d = Array2::<f64>::zeros((n_stars, n_groups)); // C vs F array?
                                                                 //
    let lamdaopt_sigma2: Array1<f64> = lamdaopt.map(|&x| x * x * sigmaopt * sigmaopt);
    let w_inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(n_dim as i32))) * weights;

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_stars / n_threads, 50_000), 10_000);
    // println!("New parallel reverse fixed : {}", n_chunk);
    // println!("threads: {}", n_threads);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos_2d.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_2d_small)| {
                // let mut stars_kdtree: KdTree<f64, 3> = KdTree::with_capacity(n_chunk);
                let mut stars_kdtree: float_KdTree<f64, usize, 3, 128, u32> =
                    float_KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt_sigma2)
                    .and(&w_inv_lamdaopt_pow)
                    .and(&group_inds)
                    .for_each(|p_row, lamda_s2, w_inv_lamda, g_ind| {
                        let neighbours = stars_kdtree.within_unsorted(
                            ndarray_to_array3(p_row.to_slice().unwrap()),
                            *lamda_s2,
                            &squared_euclidean,
                        );
                        for neigh in neighbours {
                            let t_2 = neigh.distance / lamda_s2;
                            rhos_2d_small[(neigh.item, *g_ind)] += (1. - t_2) * w_inv_lamda;
                        }
                    });
            });
    });

    //
    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos_2d *= sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd); //can
                                                                                 // *(1. / n_points as f64)
                                                                                 //include the sigmaopt earlier
    rhos_2d
}

pub fn epanechnikov_density_kde_3d_rev_weights_multi(
    x: ArrayView2<f64>,
    multi_points: Vec<ArrayView2<f64>>,
    multi_lamdaopt: Vec<ArrayView1<f64>>,
    multi_weights: Vec<ArrayView1<f64>>,
    multi_sigmaopt: Vec<f64>,
    n_threads: usize,
) -> Array2<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!

    let n_groups = multi_points.len();

    let mut rhos_2d = Array2::<f64>::zeros((n_stars, n_groups)); // C vs F array?

    let mut multi_lamdaopt_sigma2: Vec<Array1<f64>> = Vec::new();
    let mut multi_w_inv_lamdaopt_pow: Vec<Array1<f64>> = Vec::new();

    for (points, lamdaopt, weights, sigmaopt) in izip!(
        multi_points.iter(),
        multi_lamdaopt.iter(),
        multi_weights.iter(),
        multi_sigmaopt.iter()
    ) {
        let points_shape = points.shape();
        let n_dim_points: usize = points_shape[1];
        assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!
        multi_lamdaopt_sigma2.push(lamdaopt.map(|&x| x * x * sigmaopt * sigmaopt));
        multi_w_inv_lamdaopt_pow.push(lamdaopt.map(|&x| x.powi(-(n_dim as i32))) * weights);
    }

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_stars / n_threads, 50_000), 10_000);
    // println!("New parallel reverse fixed : {}", n_chunk);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos_2d.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_2d_small)| {
                let mut stars_kdtree: KdTree<f64, 3> = KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
                }

                for (mut rhos_small, points, lamdaopt_sigma2, w_inv_lamdaopt_pow, sigmaopt) in izip!(
                    rhos_2d_small.axis_iter_mut(Axis(1)),
                    multi_points.iter(),
                    multi_lamdaopt_sigma2.iter(),
                    multi_w_inv_lamdaopt_pow.iter(),
                    multi_sigmaopt.iter()
                ) {
                    Zip::from(points.axis_iter(Axis(0)))
                        .and(lamdaopt_sigma2)
                        .and(w_inv_lamdaopt_pow)
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

                    // rhos_small *= (1. / points.shape()[0] as f64) * sigmaopt.powi(-(n_dim as i32))
                    rhos_small *=  sigmaopt.powi(-(n_dim as i32))
                }
            });
    });

    //
    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos_2d *= (n_dim as f64 + 2.) / (2. * vd); //can
                                                //include the sigmaopt earlier
    rhos_2d
}

pub fn epanechnikov_density_kde_2d_rev_weights_groups(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    group_inds: ArrayView1<usize>,
    n_groups: usize,
    sigmaopt: f64,
    n_threads: usize,
) -> Array2<f64> {
    let x_shape = x.shape();
    let n_stars: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    let points_shape = points.shape();
    // let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];
    assert_eq!(n_dim, 2); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 2); //else ndarray to array3 is not allowed!

    let mut rhos_2d = Array2::<f64>::zeros((n_stars, n_groups)); // C vs F array?
                                                                 //
    let lamdaopt_sigma2: Array1<f64> = lamdaopt.map(|&x| x * x * sigmaopt * sigmaopt);
    let w_inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(n_dim as i32))) * weights;

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_stars / n_threads, 50_000), 10_000);
    // println!("New parallel reverse fixed : {}", n_chunk);
    // println!("threads: {}", n_threads);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos_2d.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_2d_small)| {
                let mut stars_kdtree: KdTree<f64, 2> = KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(ndarray_to_array2(jvec.to_slice().unwrap()), idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt_sigma2)
                    .and(&w_inv_lamdaopt_pow)
                    .and(&group_inds)
                    .for_each(|p_row, lamda_s2, w_inv_lamda, g_ind| {
                        let neighbours = stars_kdtree.within_unsorted(
                            ndarray_to_array2(p_row.to_slice().unwrap()),
                            *lamda_s2,
                            &squared_euclidean,
                        );
                        for neigh in neighbours {
                            let t_2 = neigh.distance / lamda_s2;
                            rhos_2d_small[(neigh.item, *g_ind)] += (1. - t_2) * w_inv_lamda;
                        }
                    });
            });
    });

    //
    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos_2d *= sigmaopt.powi(-(n_dim as i32)) * (n_dim as f64 + 2.) / (2. * vd); //can
                                                                                 // *(1. / n_points as f64)
                                                                                 //include the sigmaopt earlier
    rhos_2d
}
