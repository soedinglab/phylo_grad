#![allow(non_snake_case, clippy::needless_range_loop)]

extern crate nalgebra as na;

use num_traits::Float;
use numpy::ndarray::{Array, ArrayView1, ArrayView2, ArrayView3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use phylo_grad::FloatTrait;
use pyo3::{
    pyclass, pymethods, pymodule, types::PyModule, Bound, IntoPy, PyObject, PyRefMut, PyResult,
    Python,
};

use std::collections::HashMap;

use phylo_grad::{FelsensteinResult, FelsensteinTree};

pub fn backend_from_py<F: FloatTrait + numpy::Element, const DIM: usize>(
    tree: PyReadonlyArray2<'_, F>,
    leaf_log_p: PyReadonlyArray3<'_, F>,
    distance_threshold: F,
) -> FelsensteinTree<F, DIM> {
    let (parents, distances) = array2tree(tree, distance_threshold);
    let leaf_log_p = leaf_log_p.as_array();
    let leaf_log_p_shape = leaf_log_p.shape(); // [L, num_leaves, DIM]
    assert!(DIM == leaf_log_p_shape[2]);
    let leaf_log_p = vec_leaf_p_from_python(leaf_log_p);

    let mut tree = FelsensteinTree::new(&parents, &distances);
    tree.bind_leaf_log_p(leaf_log_p);
    tree
}

pub fn backend_calc_grad_py<F: FloatTrait + numpy::Element, const DIM: usize>(
    backend: &mut FelsensteinTree<F, DIM>,
    s: PyReadonlyArray3<'_, F>,
    sqrt_pi: PyReadonlyArray2<'_, F>,
) -> FelsensteinResult<F, DIM> {
    let s = vec_2d_from_python(s);
    let sqrt_pi = vec_1d_from_python(sqrt_pi);
    backend.calculate_gradients(&s, &sqrt_pi)
}

fn vec_0d_into_python<T>(vec: Vec<T>, py: Python) -> Bound<PyArray1<T>>
where
    T: numpy::Element,
{
    Array::<T, _>::from_shape_vec((vec.len(),), vec)
        .unwrap()
        .into_pyarray_bound(py)
}

fn na_1d_from_python<T, const N: usize>(py_array1: ArrayView1<'_, T>) -> na::SVector<T, N>
where
    T: numpy::Element + na::Scalar + Copy,
{
    na::SVector::<T, N>::from_iterator(py_array1.iter().copied())
}

fn vec_1d_from_python<T, const N: usize>(
    py_array2: PyReadonlyArray2<'_, T>,
) -> Vec<na::SVector<T, N>>
where
    T: numpy::Element + na::Scalar + Copy,
{
    let ndarray = py_array2.as_array();
    let vec: Vec<na::SVector<T, N>> = ndarray.axis_iter(Axis(0)).map(na_1d_from_python).collect();
    vec
}

fn vec_1d_into_python<T, const N: usize>(
    vec: Vec<na::SVector<T, N>>,
    py: Python<'_>,
) -> Bound<'_, PyArray2<T>>
where
    T: numpy::Element + na::Scalar + Copy,
{
    Array::<T, _>::from_shape_vec(
        (vec.len(), N),
        vec.into_iter()
            .flat_map(|vector| vector.iter().copied().collect::<Vec<T>>())
            .collect::<Vec<T>>(),
    )
    .unwrap()
    .into_pyarray_bound(py)
}

fn na_2d_from_python<T, const R: usize, const C: usize>(
    py_array2: ArrayView2<'_, T>,
) -> na::SMatrix<T, R, C>
where
    T: numpy::Element + na::Scalar + Copy,
{
    na::SMatrix::<T, R, C>::from_iterator(py_array2.t().iter().copied())
}

fn vec_2d_from_python<T, const R: usize, const C: usize>(
    py_array3: PyReadonlyArray3<'_, T>,
) -> Vec<na::SMatrix<T, R, C>>
where
    T: numpy::Element + na::Scalar + Copy,
{
    let ndarray = py_array3.as_array();
    let vec: Vec<na::SMatrix<T, R, C>> =
        ndarray.axis_iter(Axis(0)).map(na_2d_from_python).collect();
    vec
}

fn vec_2d_into_python<T, const R: usize, const C: usize>(
    vec: Vec<na::SMatrix<T, R, C>>,
    py: Python<'_>,
) -> Bound<'_, PyArray3<T>>
where
    T: numpy::Element + na::Scalar + Copy,
{
    Array::<T, _>::from_shape_vec(
        (vec.len(), R, C),
        vec.into_iter()
            .flat_map(|matrix| matrix.transpose().into_iter().copied().collect::<Vec<T>>())
            .collect::<Vec<T>>(),
    )
    .unwrap()
    .into_pyarray_bound(py)
}

fn to_vec_DIM<F: FloatTrait + numpy::Element, const DIM: usize>(
    py_array2: ArrayView2<'_, F>,
) -> Vec<na::SVector<F, DIM>> {
    py_array2
        .axis_iter(Axis(0))
        .map(|py_array1| na_1d_from_python::<F, DIM>(py_array1))
        .collect()
}

fn vec_leaf_p_from_python<F: FloatTrait + numpy::Element, const DIM: usize>(
    py_array3: ArrayView3<F>,
) -> Vec<Vec<na::SVector<F, DIM>>> {
    let ndarray = py_array3;
    let vec: Vec<Vec<na::SVector<F, DIM>>> = ndarray
        .axis_iter(Axis(0))
        .map(|nodes_axis| to_vec_DIM(nodes_axis))
        .collect();
    vec
}

fn inference_into_py<F: FloatTrait + numpy::Element, const DIM: usize>(
    result: FelsensteinResult<F, DIM>,
    py: Python<'_>,
) -> PyObject {
    let log_likelihood_total_py = vec_0d_into_python(result.log_likelihood, py);
    let grad_s_total_py = vec_2d_into_python(result.grad_s, py);
    let grad_sqrt_pi_total_py = vec_1d_into_python(result.grad_sqrt_pi, py);

    let result: HashMap<String, PyObject> = [
        ("log_likelihood".to_string(), log_likelihood_total_py.into()),
        ("grad_s".to_string(), grad_s_total_py.into()),
        ("grad_sqrt_pi".to_string(), grad_sqrt_pi_total_py.into()),
    ]
    .into_iter()
    .collect();

    result.into_py(py)
}

fn array2tree<F: FloatTrait + numpy::Element>(
    tree: PyReadonlyArray2<'_, F>,
    distance_threshold: F,
) -> (Vec<i32>, Vec<F>) {
    let tree = tree.as_array();
    assert!(tree.shape()[1] == 2);
    let distances = tree
        .column(1)
        .map(|x| Float::max(*x, distance_threshold))
        .to_vec();
    let parents = tree
        .column(0)
        .map(|x| {
            x.to_i32()
                .expect("Tree parent ids should be integers fitting into i32")
        })
        .to_vec();

    (parents, distances)
}

macro_rules! backend_both {
    ($float:ty, $dim:expr) => {
        paste::item! {
            #[pyclass]
            #[allow(non_camel_case_types)]
            struct [<Backend_ $float _ $dim>] {
                tree: FelsensteinTree<$float, $dim>,
            }

            #[pymethods]
            impl [<Backend_ $float _ $dim>] {
                #[new]
                fn py_new(
                    tree: PyReadonlyArray2<$float>,
                    leaf_log_p: PyReadonlyArray3<$float>,
                    distance_threshold: $float,
                ) -> PyResult<Self> {
                    Ok([<Backend_ $float _ $dim>] {
                        tree: backend_from_py(tree, leaf_log_p, distance_threshold),
                    })
                }

                #[pyo3(signature=(s, sqrt_pi))]
                fn calculate_gradients(
                    mut self_: PyRefMut<'_, Self>,
                    s: PyReadonlyArray3<$float>,
                    sqrt_pi: PyReadonlyArray2<$float>,
                ) -> PyResult<PyObject> {
                    let py = self_.py();
                    let backend = &mut self_.tree;

                    Ok(inference_into_py(backend_calc_grad_py(backend, s, sqrt_pi), py))
                }
            }
        }
    };
}

macro_rules! backend {
    ($dim:expr) => {
        backend_both!(f32, $dim);
        backend_both!(f64, $dim);
    };
}

macro_rules! backend_all {
    ($($dim:expr), *) => {
        $(
            backend!($dim);
        )*
    };
}

macro_rules! add_class {
    ($mod:expr, $dim:expr) => {
        paste::item! {
            $mod.add_class::<[<Backend_f32_ $dim>]>()?;
            $mod.add_class::<[<Backend_f64_ $dim>]>()?;
        }
    };
}

macro_rules! add_class_all {
    ($mod:expr, $($dim:expr),*) => {
        $(
            add_class!($mod, $dim);
        )*
    };
}

backend_all!(4, 16, 20);

#[pymodule]
#[pyo3(name = "_phylo_grad")]
fn phylo_grad_mod<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    add_class_all!(m, 4, 16, 20);
    Ok(())
}
