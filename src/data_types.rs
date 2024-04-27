use crate::ndarray::prelude::*;
use crate::num_enum::{IntoPrimitive, TryFromPrimitive};

#[derive(Debug, Copy, Clone, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Residue {
    None = 0,
    A,
    C,
    G,
    T,
}

impl Residue {
    /* TODO: magic value */
    pub const TOTAL: usize = 5;
}

/* TODO #[repr(u8)] */
#[derive(Debug)]
struct ResiduePair(Residue, Residue);

pub type Entry = Residue;
/* TODO: f32 not allowed by expm crate. That crate is only used for debugging purposes though. */
pub type Float = f64;

#[derive(Debug, Clone, Copy)]
pub struct FelsensteinNodeStd<const TOTAL: usize> {
    pub(crate) log_p: [Float; TOTAL],
    pub(crate) distance: Float,
}
#[derive(Debug)]
pub struct FelsensteinNodeNdarray {
    pub(crate) log_p: Array1<Float>,
    pub(crate) distance: Float,
}

pub fn log_indicator<const TOTAL: usize>(entry: Entry) -> [Float; TOTAL] {
    let mut arr = [Float::NEG_INFINITY; TOTAL];
    arr[entry as usize] = 0 as Float;
    arr
}

pub fn log_indicator_ndarray(entry: Entry) -> Array1<Float> {
    let mut arr = Array1::from_elem((Entry::TOTAL,), Float::NEG_INFINITY);
    arr[entry as usize] = 0 as Float;
    arr
}

impl<const TOTAL: usize> From<Entry> for FelsensteinNodeStd<TOTAL> {
    fn from(entry: Entry) -> Self {
        FelsensteinNodeStd {
            log_p: log_indicator(entry),
            distance: 1.0,
        }
    }
}

impl From<Entry> for FelsensteinNodeNdarray {
    fn from(entry: Entry) -> Self {
        FelsensteinNodeNdarray {
            log_p: log_indicator_ndarray(entry),
            distance: 1.0,
        }
    }
}

pub type FelsensteinNode = FelsensteinNodeNdarray;
