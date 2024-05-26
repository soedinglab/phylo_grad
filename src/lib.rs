#![allow(clippy::needless_range_loop)]
extern crate nalgebra as na;

use numpy::ndarray::{Array, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods, pymodule,
    types::{PyModule, PyString},
    Bound, PyRef, PyResult, Python,
};

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

/* TODO rename ResidueExtended to Residue4 */
/* TODO handle generics (can we still have a generic FTreeBackend?) */
//type Residue = ResidueExtended;
//type Entry = ResiduePair<ResidueExtended>;

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

    fn infer(
        &self,
        index_pairs: &Vec<(usize, usize)>,
        /* TODO as_deref() */
        rate_matrices: &[na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>],
        log_p_priors: &[[Float; Entry::DIM]],
    ) -> (
        Vec<Float>,
        Vec<na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>>,
        Vec<[Float; Entry::DIM]>,
    ) {
        train_parallel(
            index_pairs,
            self.residue_sequences_2d.as_view(),
            rate_matrices,
            log_p_priors,
            &(self.tree),
            &(self.distances),
        )
    }
}

/* TODO remove: exists for debugging purposes */
#[pymethods]
impl FTreeBackend {
    fn debug<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        PyString::new_bound(py, &format!("{:?}", self.tree))
    }
}

/* struct InferenceResult {
    log_likelihood_total:
    grad_rate_total:
    grad_log_prior_total:
} */

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

    /* TODO terrible, copies everywhere */
    //#[pyo3(signature = (index_pairs, rate_matrix, log_p_prior))]
    fn infer<'py>(
        self_: PyRef<'py, Self>,
        index_pairs: PyReadonlyArray2<'py, usize>,
        rate_matrices: PyReadonlyArray3<'py, Float>,
        log_p_priors: PyReadonlyArray2<'py, Float>,
    ) -> (
        Bound<'py, PyArray1<Float>>,
        Bound<'py, PyArray3<Float>>,
        Bound<'py, PyArray2<Float>>,
    ) {
        let super_ = self_.as_ref();
        let py = self_.py();

        let index_pairs_ndarray = index_pairs.as_array();
        let index_pairs_vec: Vec<(_, _)> = index_pairs_ndarray
            .axis_iter(Axis(0))
            .map(|col| (col[0], col[1]))
            .collect();

        let rate_matrices_ndarray = rate_matrices.as_array();
        let rate_matrices_vec: Vec<na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>> =
            rate_matrices_ndarray
                .axis_iter(Axis(0))
                .map(|slice_2d| {
                    na::SMatrix::<Float, { Entry::DIM }, { Entry::DIM }>::from_iterator(
                        slice_2d.t().iter().copied(),
                    )
                })
                .collect();

        let log_p_priors_ndarray = log_p_priors.as_array();
        let log_p_priors_vec: Vec<[Float; Entry::DIM]> = log_p_priors_ndarray
            .axis_iter(Axis(0))
            .map(|slice| slice.as_slice().unwrap().try_into().unwrap())
            .collect();

        let log_likelihood_total: Vec<Float>;
        let grad_rate_total: Vec<na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>>;
        let grad_log_prior_total: Vec<[Float; Entry::DIM]>;
        (log_likelihood_total, grad_rate_total, grad_log_prior_total) =
            super_.infer(&index_pairs_vec, &rate_matrices_vec, &log_p_priors_vec);

        let log_likelihood_total_py =
            Array::<Float, _>::from_shape_vec((log_likelihood_total.len(),), log_likelihood_total)
                .unwrap()
                .into_pyarray_bound(py);
        let grad_rate_total_py = Array::<Float, _>::from_shape_vec(
            (grad_rate_total.len(), Entry::DIM, Entry::DIM),
            grad_rate_total
                .into_iter()
                .flat_map(|matrix| {
                    matrix
                        .transpose()
                        .into_iter()
                        .copied()
                        .collect::<Vec<Float>>()
                })
                .collect::<Vec<Float>>(),
        )
        .unwrap()
        .into_pyarray_bound(py);
        let grad_log_prior_total_py = Array::<Float, _>::from_shape_vec(
            (grad_log_prior_total.len(), Entry::DIM),
            grad_log_prior_total
                .into_iter()
                .flat_map(|array| IntoIterator::into_iter(array))
                .collect::<Vec<Float>>(),
        )
        .unwrap()
        .into_pyarray_bound(py);

        (
            log_likelihood_total_py,
            grad_rate_total_py,
            grad_log_prior_total_py,
        )
    }
}

#[pymodule]
fn felsenstein_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<FTree>()?;
    Ok(())
}
