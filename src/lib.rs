#![allow(non_snake_case, clippy::needless_range_loop)]
extern crate nalgebra as na;

use num_traits::Float;
use numpy::ndarray::{Array, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{
    exceptions::PyValueError, pyclass, pymethods, pymodule, types::PyModule, Bound, IntoPy,
    PyObject, PyRef, PyResult, Python,
};

use std::collections::HashMap;

mod backward;
mod data_types;
mod forward;
mod preprocessing;
mod train;
mod tree;

use crate::data_types::*;
use crate::preprocessing::*;
use crate::train::*;
use crate::tree::*;

struct FTreeBackend<F, const DIM: usize> {
    tree: Vec<TreeNodeId<u32>>,
    distances: Vec<F>,
    leaf_log_p: Vec<Vec<na::SVector<F, DIM>>>,
}

impl<F: FloatTrait, const DIM : usize> FTreeBackend<F, DIM> {
    pub fn new(tree: PyReadonlyArray2<'_, F>, leaf_log_p: PyReadonlyArray3<'_, F>, distance_threshold : F) -> Self {
        let (parents, distances) = array2tree(tree, distance_threshold);
        let leaf_log_p = leaf_log_p.as_array();
        let leaf_log_p_shape = leaf_log_p.shape(); // [L num_leaves, DIM]
        assert!(DIM == leaf_log_p_shape[2]);
        let (tree, distances) = topological_preprocess::<F>(parents, distances, leaf_log_p_shape[1] as u32).expect("Tree topology is invalid");
        let leaf_log_p = vec_leaf_p_from_python(leaf_log_p);

        dbg!(leaf_log_p.len());
        dbg!(leaf_log_p[0].len());

        FTreeBackend {
            tree,
            distances,
            leaf_log_p,
        }        
    }

    pub fn infer(&self, s: PyReadonlyArray3<'_, F>, sqrt_pi: PyReadonlyArray2<'_, F>) -> InferenceResultParam<F, DIM> {
        let s_vec: Vec<na::SMatrix<F, DIM, DIM>> = vec_2d_from_python(s);
        let sqrt_pi_vec: Vec<na::SVector<F, DIM>> = vec_1d_from_python(sqrt_pi);

        train_parallel_param_unpaired(&self.leaf_log_p, &s_vec, &sqrt_pi_vec, &self.tree, &self.distances)
    }
}

enum FTreeBackendSingle {
    K4(FTreeBackend<f32, 4>),
    K5(FTreeBackend<f32, 5>),
    K16(FTreeBackend<f32, 16>),
    K20(FTreeBackend<f32, 20>),
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
    py_array3: ArrayView3<F>,
) -> Vec<Vec<na::SVector<F, DIM>>> {
    let ndarray = py_array3;
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

fn array2tree<F: FloatTrait>(tree: PyReadonlyArray2<'_, F>, distance_threshold : F) -> (Vec<i32>, Vec<F>) {
    let tree = tree.as_array();
    assert!(tree.shape()[1] == 2);
    let distances = tree.column(1).map(|x| Float::max(*x, distance_threshold)).to_vec();
    let parents = tree.column(0).map(|x| x.to_i32().expect("Tree parent ids should be integers fitting into i32")).to_vec();
    
    (parents, distances)
}

#[pyclass]
struct FTreeDouble {
    backend: FTreeBackendDouble,
}

#[pymethods]
impl FTreeDouble {
    #[new]
    fn py_new(
        k : u32,
        tree: PyReadonlyArray2<'_, f64>,
        leaf_log_p: PyReadonlyArray3<'_, f64>,
        distance_threshold: f64,
    ) -> PyResult<Self> {
        if k == 4 {
            Ok(FTreeDouble {
                backend: FTreeBackendDouble::K4(FTreeBackend::new(tree, leaf_log_p, distance_threshold)),
            })
        } else if k == 5 {
            Ok(FTreeDouble {
                backend: FTreeBackendDouble::K5(FTreeBackend::new(tree, leaf_log_p, distance_threshold)),
            })
        } else if k == 16 {
            Ok(FTreeDouble {
                backend: FTreeBackendDouble::K16(FTreeBackend::new(tree, leaf_log_p, distance_threshold)),
            })
        } else if k == 20 {
            Ok(FTreeDouble {
                backend: FTreeBackendDouble::K20(FTreeBackend::new(tree, leaf_log_p, distance_threshold)),
            })
        } else {
            Err(PyValueError::new_err("unsupported k value"))
        }
    }

    #[pyo3(signature=(s, sqrt_pi))]
    fn infer_param_unpaired<'py>(
        self_: PyRef<'py, Self>,
        s: PyReadonlyArray3<'py, f64>,
        sqrt_pi: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyObject> {
        let backend = &self_.backend;
        let py = self_.py();

        match backend {
            FTreeBackendDouble::K4(backend) => Ok(backend.infer(s, sqrt_pi).into_py(py)),
            FTreeBackendDouble::K5(backend) => Ok(backend.infer(s, sqrt_pi).into_py(py)),
            FTreeBackendDouble::K16(backend) => Ok(backend.infer(s, sqrt_pi).into_py(py)),
            FTreeBackendDouble::K20(backend) => Ok(backend.infer(s, sqrt_pi).into_py(py)),
        }
    }
}
#[pyclass]
struct FTreeSingle {
    backend: FTreeBackendSingle,
}

#[pymethods]
impl FTreeSingle {
    #[new]
    fn py_new(
        k : u32,
        tree: PyReadonlyArray2<'_, f32>,
        leaf_log_p: PyReadonlyArray3<'_, f32>,
        distance_threshold: f32,
    ) -> PyResult<Self> {
        if k == 4 {
            Ok(FTreeSingle {
                backend: FTreeBackendSingle::K4(FTreeBackend::new(tree, leaf_log_p, distance_threshold)),
            })
        } else if k == 5 {
            Ok(FTreeSingle {
                backend: FTreeBackendSingle::K5(FTreeBackend::new(tree, leaf_log_p, distance_threshold)),
            })
        } else if k == 16 {
            Ok(FTreeSingle {
                backend: FTreeBackendSingle::K16(FTreeBackend::new(tree, leaf_log_p, distance_threshold)),
            })
        } else if k == 20 {
            Ok(FTreeSingle {
                backend: FTreeBackendSingle::K20(FTreeBackend::new(tree, leaf_log_p, distance_threshold)),
            })
        } else {
            Err(PyValueError::new_err("unsupported k value"))
        }
    }

    #[pyo3(signature=(s, sqrt_pi))]
    fn infer_param_unpaired<'py>(
        self_: PyRef<'py, Self>,
        s: PyReadonlyArray3<'py, f32>,
        sqrt_pi: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<PyObject> {
        let backend = &self_.backend;
        let py = self_.py();

        match backend {
            FTreeBackendSingle::K4(backend) => Ok(backend.infer(s, sqrt_pi).into_py(py)),
            FTreeBackendSingle::K5(backend) => Ok(backend.infer(s, sqrt_pi).into_py(py)),
            FTreeBackendSingle::K16(backend) => Ok(backend.infer(s, sqrt_pi).into_py(py)),
            FTreeBackendSingle::K20(backend) => Ok(backend.infer(s, sqrt_pi).into_py(py)),
        }
    }
}

#[pymodule]
fn felsenstein_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<FTreeSingle>()?;
    m.add_class::<FTreeDouble>()?;
    Ok(())
}
