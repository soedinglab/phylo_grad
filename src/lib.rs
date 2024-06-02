#![allow(dead_code, non_snake_case, clippy::needless_range_loop)]
extern crate nalgebra as na;

use numpy::ndarray::{Array, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
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

/* TODO handle generics (can we still have a generic FTreeBackend?) */
type Entry = ResiduePair<Residue>;
//type Float = f64;

/* TODO! FTreeBackend can't know what residue type to use. It should store String,
and infer() may then call preprocessing::try_residue_sequences_from_strings */
#[pyclass(subclass)]
struct FTreeBackend {
    tree: Vec<TreeNodeId<usize>>,
    distances: Vec<Float>,
    residue_sequences_2d: na::DMatrix<Residue>,
}

impl FTreeBackend {
    fn try_from_file(data_path: &str, distance_threshold: Float) -> Result<Self, Box<dyn Error>> {
        let mut record_reader = read_raw_csv(data_path, 1)?;
        let raw_tree = deserialize_raw_tree(&mut record_reader)?;

        let tree;
        let mut distances;
        let sequences_raw;
        (tree, distances, sequences_raw) = preprocess_weak(&raw_tree)?;

        distances
            .iter_mut()
            .for_each(|d| *d = distance_threshold.max(*d));

        let residue_sequences_2d = try_residue_sequences_from_strings(&sequences_raw)?;

        Ok(Self {
            tree,
            distances,
            residue_sequences_2d,
        })
    }
    fn infer<const DIM: usize, D>(
        &self,
        index_pairs: &Vec<(usize, usize)>,
        /* TODO as_deref() */
        rate_matrices: &[na::SMatrix<Float, DIM, DIM>],
        log_p_priors: &[na::SVector<Float, DIM>],
        distributions: &[D],
    ) -> InferenceResult<Float, DIM>
    where
        D: Distribution<Entry, Float, DIM>,
        //ResiduePair<Residue>: EntryTrait<LogPType = na::SVector<Floatloat, DIM>>,
        na::Const<DIM>: Doubleable + Exponentiable,
        TwoTimesConst<DIM>: Exponentiable,
        na::DefaultAllocator: ViableAllocator<Float, DIM>,
    {
        train_parallel(
            index_pairs,
            self.residue_sequences_2d.as_view(),
            rate_matrices,
            log_p_priors,
            &(self.tree),
            &(self.distances),
            distributions,
        )
    }

    fn infer_param<const DIM: usize, D>(
        &self,
        index_pairs: &Vec<(usize, usize)>,
        /* TODO as_deref() */
        deltas: &[na::SMatrix<Float, DIM, DIM>],
        sqrt_pi: &[na::SVector<Float, DIM>],
        distributions: &[D],
    ) -> InferenceResultParam<Float, DIM>
    where
        D: Distribution<Entry, Float, DIM>,
    {
        train_parallel_param(
            index_pairs,
            self.residue_sequences_2d.as_view(),
            deltas,
            sqrt_pi,
            &(self.tree),
            &(self.distances),
            distributions,
        )
    }
}

/* struct InferenceResult {
    log_likelihood_total:
    grad_rate_total:
    grad_log_prior_total:
} */

fn vec_0d_from_python<'py, T>(py_array1: PyReadonlyArray1<'py, T>) -> Vec<T>
where
    T: numpy::Element,
{
    let ndarray = py_array1.as_array();
    ndarray.to_vec()
}

fn vec_0d_into_python<'py, T>(vec: Vec<T>, py: Python<'py>) -> Bound<PyArray1<T>>
where
    T: numpy::Element,
{
    Array::<T, _>::from_shape_vec((vec.len(),), vec)
        .unwrap()
        .into_pyarray_bound(py)
}

fn vec_1d_from_python<'py, T, const N: usize>(
    py_array2: PyReadonlyArray2<'py, T>,
) -> Vec<na::SVector<T, N>>
where
    T: numpy::Element + na::Scalar + Copy,
{
    let ndarray = py_array2.as_array();
    let vec: Vec<na::SVector<T, N>> = ndarray
        .axis_iter(Axis(0))
        .map(|slice| na::SVector::<T, N>::from_iterator(slice.iter().copied()))
        .collect();
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

fn vec_2d_from_python<'py, T, const R: usize, const C: usize>(
    py_array3: PyReadonlyArray3<'py, T>,
) -> Vec<na::SMatrix<T, R, C>>
where
    T: numpy::Element + na::Scalar + Copy,
{
    let ndarray = py_array3.as_array();
    let vec: Vec<na::SMatrix<T, R, C>> = ndarray
        .axis_iter(Axis(0))
        .map(|slice_2d| na::SMatrix::<T, R, C>::from_iterator(slice_2d.t().iter().copied()))
        .collect();
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

impl<F, const DIM: usize> IntoPy<PyObject> for InferenceResult<F, DIM>
where
    F: numpy::Element + na::Scalar + Copy,
{
    fn into_py(self, py: Python<'_>) -> PyObject {
        let log_likelihood_total_py = vec_0d_into_python(self.log_likelihood_total, py);
        let grad_rate_total_py = vec_2d_into_python(self.grad_rate_total, py);
        let grad_log_prior_total_py = vec_1d_into_python(self.grad_log_prior_total, py);

        let result: HashMap<String, PyObject> = [
            ("log_likelihood".to_string(), log_likelihood_total_py.into()),
            ("grad_rate".to_string(), grad_rate_total_py.into()),
            ("grad_log_prior".to_string(), grad_log_prior_total_py.into()),
        ]
        .into_iter()
        .collect();

        result.into_py(py)
    }
}

impl<F, const DIM: usize> IntoPy<PyObject> for InferenceResultParam<F, DIM>
where
    F: numpy::Element + na::Scalar + Copy,
{
    fn into_py<'py>(self, py: Python<'_>) -> PyObject {
        let log_likelihood_total_py = vec_0d_into_python(self.log_likelihood_total, py);
        let grad_delta_total_py = vec_2d_into_python(self.grad_delta_total, py);
        let grad_sqrt_pi_total_py = vec_1d_into_python(self.grad_sqrt_pi_total, py);
        let grad_rate_total_py = vec_2d_into_python(self.grad_rate_total, py);

        let result: HashMap<String, PyObject> = [
            ("log_likelihood".to_string(), log_likelihood_total_py.into()),
            ("grad_delta".to_string(), grad_delta_total_py.into()),
            ("grad_sqrt_pi".to_string(), grad_sqrt_pi_total_py.into()),
            ("grad_rate".to_string(), grad_rate_total_py.into()),
        ]
        .into_iter()
        .collect();

        result.into_py(py)
    }
}

#[pyclass(extends=FTreeBackend, subclass)]
struct FTree {}

#[pymethods]
impl FTree {
    #[new]
    fn py_new<'py>(data_path: &str, distance_threshold: f64) -> PyResult<(Self, FTreeBackend)> {
        let result = FTreeBackend::try_from_file(data_path, distance_threshold);
        match result {
            Ok(backend) => Ok((FTree {}, backend)),
            Err(error) => Err(PyValueError::new_err(format!(
                "Failed to create FTree: {}",
                error
            ))),
        }
    }

    #[pyo3(signature=(index_pairs, rate_matrices, log_p_priors, gaps = false, p_none = None))]
    fn infer<'py>(
        self_: PyRef<'py, Self>,
        index_pairs: PyReadonlyArray2<'py, usize>,
        rate_matrices: PyReadonlyArray3<'py, Float>,
        log_p_priors: PyReadonlyArray2<'py, Float>,
        gaps: bool,
        p_none: Option<PyReadonlyArray2<'py, Float>>,
    ) -> PyObject {
        let super_ = self_.as_ref();
        let py = self_.py();

        let index_pairs_ndarray = index_pairs.as_array();
        let index_pairs_vec: Vec<(_, _)> = index_pairs_ndarray
            .axis_iter(Axis(0))
            .map(|col| (col[0], col[1]))
            .collect();

        let result_py: PyObject;

        if !gaps {
            let distributions: Vec<DistNoGaps> = match p_none {
                Some(pyarr2) => pyarr2
                    .as_array()
                    .axis_iter(Axis(0))
                    .map(|slice| na::SVector::<f64, 4>::from_iterator(slice.iter().copied()))
                    .map(|p_none| DistNoGaps {
                        p_none: Some(p_none),
                    })
                    .collect(),
                None => std::iter::repeat(DistNoGaps { p_none: None })
                    .take(index_pairs_vec.len())
                    .collect(),
            };

            let rate_matrices_vec: Vec<na::SMatrix<f64, 16, 16>> =
                vec_2d_from_python(rate_matrices);

            let log_p_priors_vec: Vec<na::SVector<f64, 16>> = vec_1d_from_python(log_p_priors);

            let result: InferenceResult<f64, 16>;
            result = super_.infer(
                &index_pairs_vec,
                &rate_matrices_vec,
                &log_p_priors_vec,
                &distributions,
            );

            result_py = result.into_py(py);
        } else {
            let distributions: Vec<DistGaps> = std::iter::repeat(DistGaps {})
                .take(index_pairs_vec.len())
                .collect();
            let rate_matrices_vec: Vec<na::SMatrix<f64, 25, 25>> =
                vec_2d_from_python(rate_matrices);

            let log_p_priors_vec: Vec<na::SVector<f64, 25>> = vec_1d_from_python(log_p_priors);

            let result: InferenceResult<f64, 25>;
            result = super_.infer(
                &index_pairs_vec,
                &rate_matrices_vec,
                &log_p_priors_vec,
                &distributions,
            );

            result_py = result.into_py(py);
        }

        result_py
    }
    #[pyo3(signature=(index_pairs, deltas, sqrt_pi, gaps = false, p_none = None))]
    fn infer_param<'py>(
        self_: PyRef<'py, Self>,
        index_pairs: PyReadonlyArray2<'py, usize>,
        deltas: PyReadonlyArray3<'py, Float>,
        sqrt_pi: PyReadonlyArray2<'py, Float>,
        gaps: bool,
        p_none: Option<PyReadonlyArray2<'py, Float>>,
    ) -> PyObject {
        let super_ = self_.as_ref();
        let py = self_.py();

        let index_pairs_ndarray = index_pairs.as_array();
        let index_pairs_vec: Vec<(_, _)> = index_pairs_ndarray
            .axis_iter(Axis(0))
            .map(|col| (col[0], col[1]))
            .collect();

        let result_py: PyObject;
        if !gaps {
            let distributions: Vec<DistNoGaps> = match p_none {
                Some(pyarr2) => pyarr2
                    .as_array()
                    .axis_iter(Axis(0))
                    .map(|slice| na::SVector::<f64, 4>::from_iterator(slice.iter().copied()))
                    .map(|p_none| DistNoGaps {
                        p_none: Some(p_none),
                    })
                    .collect(),
                None => std::iter::repeat(DistNoGaps { p_none: None })
                    .take(index_pairs_vec.len())
                    .collect(),
            };

            let deltas_vec: Vec<na::SMatrix<f64, 16, 16>> = vec_2d_from_python(deltas);
            let sqrt_pi_vec: Vec<na::SVector<f64, 16>> = vec_1d_from_python(sqrt_pi);

            let result: InferenceResultParam<f64, 16>;
            result =
                super_.infer_param(&index_pairs_vec, &deltas_vec, &sqrt_pi_vec, &distributions);
            result_py = result.into_py(py);
        } else {
            let distributions: Vec<DistGaps> = std::iter::repeat(DistGaps {})
                .take(index_pairs_vec.len())
                .collect();

            let deltas_vec: Vec<na::SMatrix<f64, 25, 25>> = vec_2d_from_python(deltas);
            let sqrt_pi_vec: Vec<na::SVector<f64, 25>> = vec_1d_from_python(sqrt_pi);

            let result: InferenceResultParam<f64, 25>;
            result =
                super_.infer_param(&index_pairs_vec, &deltas_vec, &sqrt_pi_vec, &distributions);
            result_py = result.into_py(py);
        }
        result_py
    }
}

#[pymodule]
fn felsenstein_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<FTree>()?;
    Ok(())
}
