#![allow(non_snake_case, clippy::needless_range_loop)]
extern crate nalgebra as na;

use numpy::ndarray::{Array, ArrayView1, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{
    exceptions::PyValueError, pyclass, pymethods, pymodule, types::PyModule, Bound, IntoPy,
    PyObject, PyRef, PyResult, Python,
};

use std::collections::HashMap;
use std::error::Error;

mod backward;
mod data_types;
mod forward;
mod io;
mod preprocessing;
mod train;
mod tree;

use crate::data_types::*;
use crate::io::*;
use crate::preprocessing::*;
use crate::train::*;
use crate::tree::*;

struct FTreeBackend<F, const DIM: usize> {
    tree: Vec<TreeNodeId<usize>>,
    distances: Vec<F>,
    leaf_log_p: Vec<Vec<na::SVector<F, DIM>>>,
}

enum FTreeBackendSingle {
    K4(FTreeBackend<f32, 4>),
    K5(FTreeBackend<f64, 5>),
    K16(FTreeBackend<f64, 16>),
    K20(FTreeBackend<f64, 20>),
}

enum FTreeBackendDouble {
    K4(FTreeBackend<f64, 4>),
    K5(FTreeBackend<f64, 5>),
    K16(FTreeBackend<f64, 16>),
    K20(FTreeBackend<f64, 20>),
}

fn vec_0d_into_python<'py, T>(vec: Vec<T>, py: Python<'py>) -> Bound<PyArray1<T>>
where
    T: numpy::Element,
{
    Array::<T, _>::from_shape_vec((vec.len(),), vec)
        .unwrap()
        .into_pyarray_bound(py)
}

fn na_1d_from_python<'py, T, const N: usize>(py_array1: ArrayView1<'py, T>) -> na::SVector<T, N>
where
    T: numpy::Element + na::Scalar + Copy,
{
    na::SVector::<T, N>::from_iterator(py_array1.iter().copied())
}

fn vec_1d_from_python<'py, T, const N: usize>(
    py_array2: PyReadonlyArray2<'py, T>,
) -> Vec<na::SVector<T, N>>
where
    T: numpy::Element + na::Scalar + Copy,
{
    let ndarray = py_array2.as_array();
    let vec: Vec<na::SVector<T, N>> = ndarray.axis_iter(Axis(0)).map(na_1d_from_python).collect();
    vec
}

fn vec_1d_into_python<'py, T, const N: usize>(
    vec: Vec<na::SVector<T, N>>,
    py: Python<'py>,
) -> Bound<'py, PyArray2<T>>
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

fn na_2d_from_python<'py, T, const R: usize, const C: usize>(
    py_array2: ArrayView2<'py, T>,
) -> na::SMatrix<T, R, C>
where
    T: numpy::Element + na::Scalar + Copy,
{
    na::SMatrix::<T, R, C>::from_iterator(py_array2.t().iter().copied())
}

fn vec_2d_from_python<'py, T, const R: usize, const C: usize>(
    py_array3: PyReadonlyArray3<'py, T>,
) -> Vec<na::SMatrix<T, R, C>>
where
    T: numpy::Element + na::Scalar + Copy,
{
    let ndarray = py_array3.as_array();
    let vec: Vec<na::SMatrix<T, R, C>> =
        ndarray.axis_iter(Axis(0)).map(na_2d_from_python).collect();
    vec
}

fn vec_2d_into_python<'py, T, const R: usize, const C: usize>(
    vec: Vec<na::SMatrix<T, R, C>>,
    py: Python<'py>,
) -> Bound<'py, PyArray3<T>>
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

fn to_vec_DIM<F: FloatTrait, const DIM: usize>(
    py_array2: ArrayView2<'_, F>,
) -> Vec<na::SVector<F, DIM>> {
    py_array2
        .axis_iter(Axis(0))
        .map(|py_array1| na_1d_from_python::<F, DIM>(py_array1))
        .collect()
}
fn vec_leaf_p_from_python<'py, F: FloatTrait, const DIM: usize>(
    py_array3: PyReadonlyArray3<'py, F>,
) -> Vec<Vec<na::SVector<F, DIM>>> {
    let ndarray = py_array3.as_array();
    let vec: Vec<Vec<na::SVector<F, DIM>>> = ndarray
        .axis_iter(Axis(0))
        .map(|nodes_axis| to_vec_DIM(nodes_axis))
        .collect();
    vec
}

impl<F: FloatTrait, const DIM: usize> IntoPy<PyObject> for InferenceResultParam<F, DIM> {
    fn into_py<'py>(self, py: Python<'_>) -> PyObject {
        let log_likelihood_total_py = vec_0d_into_python(self.log_likelihood_total, py);
        let grad_delta_total_py = vec_2d_into_python(self.grad_delta_total, py);
        let grad_sqrt_pi_total_py = vec_1d_into_python(self.grad_sqrt_pi_total, py);

        let result: HashMap<String, PyObject> = [
            ("log_likelihood".to_string(), log_likelihood_total_py.into()),
            ("grad_delta".to_string(), grad_delta_total_py.into()),
            ("grad_sqrt_pi".to_string(), grad_sqrt_pi_total_py.into()),
        ]
        .into_iter()
        .collect();

        result.into_py(py)
    }
}

fn array2tree<F: FloatTrait>(tree: PyReadonlyArray2<'_, F>) -> (Vec<i32>, Vec<F>) {
    let tree = tree.as_array();
    assert!(tree.shape()[1] == 2);
    let distances = tree.column(1).to_vec();
    let parents = tree.column(0).map(|x| x.to_i32().expect("Tree parent ids should be integers fitting into i32")).to_vec();
    
    (parents, distances)
}

#[pyclass]
struct FTreeSingle {
    backend: FTreeBackendSingle,
}

#[pymethods]
impl FTreeSingle {
    #[new]
    fn py_new(
        tree: PyReadonlyArray2<'_, f32>,
        leaf_log_p: PyReadonlyArray3<'_, f32>,
        distance_threshold: f32,
    ) -> PyResult<Self> {
        todo!()
    }

    #[pyo3(signature=(leaf_log_p, s, sqrt_pi))]
    fn infer_param_unpaired<'py>(
        self_: PyRef<'py, Self>,
        leaf_log_p: PyReadonlyArray3<'py, Float>,
        s: PyReadonlyArray3<'py, Float>,
        sqrt_pi: PyReadonlyArray2<'py, Float>,
    ) -> PyResult<PyObject> {
        let backend = &self_.backend;
        let py = self_.py();

        let s_vec: Vec<na::SMatrix<Float, 4, 4>> = vec_2d_from_python(s);
        let sqrt_pi_vec: Vec<na::SVector<Float, 4>> = vec_1d_from_python(sqrt_pi);
        let leaf_log_p_vec: Vec<Vec<na::SVector<Float, 4>>> = vec_leaf_p_from_python(leaf_log_p);

        let result = train_parallel_param_unpaired(&leaf_log_p_vec, &s_vec, &sqrt_pi_vec, backend.tree, &backend.distances);

        Ok(result.into_py(py))
    }
}

#[pymodule]
fn felsenstein_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<FTree>()?;
    Ok(())
}
