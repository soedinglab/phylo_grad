#![allow(non_snake_case, clippy::needless_range_loop)]

extern crate nalgebra as na;

use numpy::ndarray::{Array, ArrayView1, ArrayView2, ArrayView3, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use phylo_grad::FloatTrait;
use pyo3::{
    pyclass, pymethods, pymodule, types::PyModule, Bound, IntoPy, PyObject, PyRefMut, PyResult,
    Python,
};

use std::collections::HashMap;

use phylo_grad::{FelsensteinResult, FelsensteinTree};

fn backend_from_py<F: FloatTrait + numpy::Element, const DIM: usize>(
    parent_list: PyReadonlyArray1<'_, i32>,
    branch_lengths: PyReadonlyArray1<'_, F>,
) -> FelsensteinTree<F, DIM> {
    let parent_list_vec = parent_list.as_array().to_vec();
    let branch_lengths_vec = branch_lengths.as_array().to_vec();

    FelsensteinTree::new(&parent_list_vec, &branch_lengths_vec)
}

fn backend_bind_leaf_log_p<F: FloatTrait + numpy::Element, const DIM: usize>(
    backend: &mut FelsensteinTree<F, DIM>,
    leaf_log_p: PyReadonlyArray3<'_, F>,
) {
    let leaf_log_p = vec_leaf_p_from_python(leaf_log_p.as_array());
    backend.bind_leaf_log_p(leaf_log_p);
}

fn backend_calc_grad_py<F: FloatTrait + numpy::Element, const DIM: usize>(
    backend: &mut FelsensteinTree<F, DIM>,
    s: PyReadonlyArray3<'_, F>,
    sqrt_pi: PyReadonlyArray2<'_, F>,
) -> FelsensteinResult<F, DIM> {
    let s = vec_2d_from_python(s);
    let sqrt_pi = vec_1d_from_python(sqrt_pi);
    backend.calculate_gradients(&s, &sqrt_pi)
}

fn backend_calc_grad_with_log_p_py<F: FloatTrait + numpy::Element, const DIM: usize>(
    backend: &FelsensteinTree<F, DIM>,
    s: PyReadonlyArray3<'_, F>,
    sqrt_pi: PyReadonlyArray2<'_, F>,
    log_p: &mut [&mut [na::SVector<F, DIM>]],
) -> FelsensteinResult<F, DIM> {
    let s = vec_2d_from_python(s);
    let sqrt_pi = vec_1d_from_python(sqrt_pi);
    backend.calculate_gradients_with_log_p(&s, &sqrt_pi, log_p)
}

fn copy_leaf_log_p_to_internal_vec<F: FloatTrait + numpy::Element, const DIM: usize>(
    vec: &mut Vec<Vec<na::SVector<F, DIM>>>,
    leaf_log_p: PyReadonlyArray3<'_, F>,
    num_nodes: usize,
) {
    let array = leaf_log_p.as_array();
    let shape = array.shape();

    vec.resize(shape[0], vec![na::SVector::<F, DIM>::zeros(); num_nodes]);

    let num_leaves = shape[1];

    assert!(shape[2] == DIM);

    for (i, log_p) in vec.iter_mut().enumerate() {
        log_p.resize(num_nodes, na::SVector::<F, DIM>::zeros());
        for j in 0..num_leaves {
            for k in 0..DIM {
                log_p[j][k] = array[[i, j, k]];
            }
        }
        for j in num_leaves..num_nodes {
            log_p[j] = na::SVector::<F, DIM>::zeros();
        }
    }
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

macro_rules! backend_both {
    ($float:ty, $dim:expr) => {
        paste::item! {
            #[pyclass]
            #[allow(non_camel_case_types)]
            struct [<Backend_ $float _ $dim>] {
                tree: FelsensteinTree<$float, $dim>,
                /// Only used for the `calculate_gradients_with_leaf_log_p` function.
                log_p: Vec<Vec<na::SVector<$float, $dim>>>,
            }

            impl [<Backend_ $float _ $dim>] {
                fn calc_grad_with_log_p(
                    &mut self,
                    s: PyReadonlyArray3<$float>,
                    sqrt_pi: PyReadonlyArray2<$float>,
                    leaf_log_p: PyReadonlyArray3<$float>,
                ) -> FelsensteinResult<$float, $dim> {
                    copy_leaf_log_p_to_internal_vec(
                        &mut self.log_p,
                        leaf_log_p,
                        self.tree.num_nodes(),
                    );
                    let mut log_p = self.log_p.iter_mut().map(|x| x.as_mut_slice()).collect::<Vec<_>>();
                    backend_calc_grad_with_log_p_py(&self.tree, s, sqrt_pi, &mut log_p)
                }
            }

            #[pymethods]
            impl [<Backend_ $float _ $dim>] {
                #[new]
                fn py_new(
                    parent_list: PyReadonlyArray1<i32>,
                    branch_lengths: PyReadonlyArray1<$float>,
                ) -> Self {
                    Self {
                        tree: backend_from_py(parent_list, branch_lengths),
                        log_p: vec![],
                    }
                }

                #[pyo3(signature=(leaf_log_p))]
                fn bind_leaf_log_p(
                    mut self_: PyRefMut<'_, Self>,
                    leaf_log_p: PyReadonlyArray3<$float>,
                ) {
                    backend_bind_leaf_log_p(&mut self_.tree, leaf_log_p);
                }

                #[pyo3(signature=(s, sqrt_pi))]
                fn calculate_gradients(
                    mut self_: PyRefMut<'_, Self>,
                    s: PyReadonlyArray3<$float>,
                    sqrt_pi: PyReadonlyArray2<$float>,
                ) -> PyObject {
                    let py = self_.py();
                    let backend = &mut self_.tree;

                    inference_into_py(backend_calc_grad_py(backend, s, sqrt_pi), py)
                }

                #[pyo3(signature=(s, sqrt_pi))]
                fn calculate_log_likelihoods<'py>(
                    mut self_: PyRefMut<'py, Self>,
                    s: PyReadonlyArray3<$float>,
                    sqrt_pi: PyReadonlyArray2<$float>,
                ) -> Bound<'py, PyArray1<$float>> {
                    let backend = &mut self_.tree;
                    let s = vec_2d_from_python(s);
                    let sqrt_pi = vec_1d_from_python(sqrt_pi);
                    let result = backend.calculate_likelihoods(&s, &sqrt_pi);
                    use numpy::ToPyArray;
                    result.to_pyarray_bound(self_.py())
                }

                #[pyo3(signature=(s, sqrt_pi))]
                fn calculate_gradients_edges<'py>(
                    mut self_: PyRefMut<'py, Self>,
                    s: PyReadonlyArray3<$float>,
                    sqrt_pi: PyReadonlyArray2<$float>,
                ) -> Bound<'py, PyArray1<$float>> {
                    let backend = &mut self_.tree;
                    let s = vec_2d_from_python(s);
                    let sqrt_pi = vec_1d_from_python(sqrt_pi);
                    let result = backend.calculate_edge_gradients(&s[0], &sqrt_pi[0]);
                    use numpy::ToPyArray;
                    result.to_pyarray_bound(self_.py())
                }
                #[pyo3(signature=(s, sqrt_pi, leaf_log_p))]
                fn calculate_gradients_with_leaf_log_p(
                    mut self_: PyRefMut<'_, Self>,
                    s: PyReadonlyArray3<$float>,
                    sqrt_pi: PyReadonlyArray2<$float>,
                    leaf_log_p: PyReadonlyArray3<$float>,
                ) -> PyObject {
                    let py = self_.py();

                    inference_into_py(self_.calc_grad_with_log_p(s, sqrt_pi, leaf_log_p), py)
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
