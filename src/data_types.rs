use std::convert::TryFrom;

use crate::itertools::Itertools;
use crate::num_enum::{IntoPrimitive, TryFromPrimitive};

pub trait Exponentiable: na::ToTypenum + na::DimMin<Self, Output = Self> {}
impl<T: na::ToTypenum + na::DimMin<Self, Output = Self>> Exponentiable for T {}

/* This does not work (and shouldn't) */
pub trait Decrementable
where
    Self: na::DimSub<na::Const<1>>,
{
}

impl<T> Decrementable for T where
    T: na::DimSub<na::Const<1>> //    na::DefaultAllocator: na::allocator::Allocator<f64, <T as na::DimSub<na::Const<1>>>::Output>,
{
}
//pub trait Doubleable

pub trait EntryTrait: Sized + Copy + na::Scalar {
    const TOTAL: usize;
    const DIM: usize;
    const CHARS: usize;
    type LogPType;
    fn to_log_p(&self) -> Self::LogPType;
    //fn try_deserialize_string(input: &str) -> Result<Vec<Self>, FelsensteinError>;
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>>;
}

#[derive(Debug, Copy, Clone, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Residue5 {
    A,
    C,
    G,
    T,
    None,
}

impl Residue5 {
    pub const TOTAL: usize = 5;
    pub const DIM: usize = 5;
}

/* TODO! either remove the u8-stuff or make it work */
#[derive(Debug, Copy, Clone, PartialEq, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum ResidueExtended {
    A,
    C,
    G,
    T,
    None,
    R,
    Y,
    S,
    W,
    K,
    M,
    B,
    D,
    H,
    V,
}
impl ResidueExtended {
    pub const U: Self = Self::T;
    pub fn columns_iter(
        residue_sequences_2d: &na::DMatrix<Self>,
    ) -> impl Iterator<Item = (usize, na::DVectorView<Self>)> {
        residue_sequences_2d.column_iter().enumerate()
    }
}

impl TryFrom<char> for ResidueExtended {
    type Error = FelsensteinError;
    fn try_from(char: char) -> Result<Self, Self::Error> {
        use ResidueExtended::*;
        let char: ResidueExtended = match char.to_ascii_uppercase() {
            'A' => A,
            'C' => C,
            'G' => G,
            'T' => T,
            'R' => R,
            'Y' => Y,
            'S' => S,
            'W' => W,
            'K' => K,
            'M' => M,
            'B' => B,
            'D' => D,
            'H' => H,
            'V' => V,
            'U' => ResidueExtended::U,
            '-' => ResidueExtended::None,
            '.' => ResidueExtended::None,
            'N' => ResidueExtended::None,
            _ => return Err(FelsensteinError::INVALID_CHAR),
        };
        Ok(char)
    }
}

impl EntryTrait for ResidueExtended {
    const TOTAL: usize = 15;
    const DIM: usize = 4;
    const CHARS: usize = 1;
    type LogPType = [Float; Self::DIM];

    fn to_log_p(&self) -> [Float; Self::DIM] {
        use ResidueExtended::*;
        let prob: [Float; 4] = match &self {
            A => [1.0, 0.0, 0.0, 0.0],
            C => [0.0, 1.0, 0.0, 0.0],
            G => [0.0, 0.0, 1.0, 0.0],
            T => [0.0, 0.0, 0.0, 1.0],
            ResidueExtended::None => [0.25, 0.25, 0.25, 0.25],
            R => [0.5, 0.0, 0.5, 0.0],
            Y => [0.0, 0.5, 0.0, 0.5],
            S => [0.0, 0.5, 0.5, 0.0],
            W => [0.5, 0.0, 0.0, 0.5],
            K => [0.0, 0.0, 0.5, 0.5],
            M => [0.5, 0.5, 0.0, 0.0],
            B => [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            D => [1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0],
            H => [1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0],
            V => [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
        };
        prob.map(Float::ln)
    }
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>> {
        input.chars().map(Self::try_from)
    }
}

/* PartialEq only needed if we want to store ResiduePair in na::Matrix */
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ResiduePair<Res>(pub Res, pub Res);

/* Can't make generic over Residue  */
impl EntryTrait for ResiduePair<ResidueExtended> {
    const DIM: usize = ResidueExtended::DIM * ResidueExtended::DIM;
    const TOTAL: usize = ResidueExtended::TOTAL * ResidueExtended::TOTAL;
    const CHARS: usize = ResidueExtended::CHARS * 2;
    type LogPType = [Float; Self::DIM];

    /* TODO! this should be static */
    fn to_log_p(&self) -> Self::LogPType {
        let mut result = [0.0 as Float; Self::DIM];
        let (first, second) = (self.0, self.1);
        let log_p_first = first.to_log_p();
        let log_p_second = second.to_log_p();
        for a in 0..ResidueExtended::DIM {
            for b in 0..ResidueExtended::DIM {
                result[ResidueExtended::DIM * a + b] = log_p_first[a] + log_p_second[b];
            }
        }
        result
    }

    /* TODO remove (replaced by pair_columns_iter) */
    /* TODO! will the external function be able to check the buffer? */
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>> {
        let tuple_iter = input.chars().tuples::<(_, _)>();
        tuple_iter.map(
            |(first, second)| -> Result<ResiduePair<ResidueExtended>, FelsensteinError> {
                Ok(ResiduePair(
                    ResidueExtended::try_from(first)?,
                    ResidueExtended::try_from(second)?,
                ))
            },
        )
    }
}

#[derive(Debug)]
pub enum FelsensteinError {
    LogicError(&'static str),
    DeserializationError(&'static str),
}

impl std::fmt::Display for FelsensteinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Self::LogicError(str) => str,
            Self::DeserializationError(str) => str,
        };
        write!(f, "{}", str)
    }
}
impl std::error::Error for FelsensteinError {}

impl FelsensteinError {
    pub const SEQ_LENGTH: Self =
        Self::DeserializationError("Sequence length not divisible by entry length");
    pub const INVALID_CHAR: Self =
        Self::DeserializationError("Invalid character in residue sequence");
    /*
    pub fn invalid_residue(s: &str) -> Self {
        Self::DeserializationError(format!("Invalid residue:{}", s))
    }*/
}

pub type Residue = ResidueExtended;
pub type Entry = ResiduePair<ResidueExtended>;
pub type Float = f64;
pub type ColumnId = usize;
pub type Id = usize;
pub type RateType = na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>;
