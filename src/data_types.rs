use std::convert::TryFrom;

use na::{
    allocator::Allocator, Const, DefaultAllocator, DimAdd, DimMin, DimMul, DimName, ToTypenum,
};
use num_enum::{IntoPrimitive, TryFromPrimitive};

pub type Float = f64;
//pub type ColumnId = usize;
pub type Id = usize;

pub const EPS_DIV: f64 = 1e-10;
pub const EPS_LOG: f64 = 1e-100;

pub trait ScalarTrait: num_traits::Float + na::Scalar {}
impl ScalarTrait for f32 {}
impl ScalarTrait for f64 {}

pub trait EntryTrait: Sized + Copy {
    const TOTAL: usize;
    const DIM: usize;
    const CHARS: usize;
    type LogPType;
    fn to_log_p(&self) -> Self::LogPType;
}
pub trait ResidueTrait: EntryTrait + na::Scalar + std::marker::Sync {
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>>;
}

/* TODO! either remove the u8-stuff or make it work */
#[derive(Debug, Copy, Clone, PartialEq, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Residue4 {
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
impl Residue4 {
    pub const U: Self = Self::T;
}

impl TryFrom<char> for Residue4 {
    type Error = FelsensteinError;
    fn try_from(char: char) -> Result<Self, Self::Error> {
        use Residue4::*;
        let res: Residue4 = match char.to_ascii_uppercase() {
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
            'U' => Residue4::U,
            '-' => Residue4::None,
            '.' => Residue4::None,
            'N' => Residue4::None,
            _ => return Err(FelsensteinError::INVALID_CHAR),
        };
        Ok(res)
    }
}

impl EntryTrait for Residue4 {
    const TOTAL: usize = 15;
    const DIM: usize = 4;
    const CHARS: usize = 1;
    type LogPType = na::SVector<Float, { Self::DIM }>;

    fn to_log_p(&self) -> Self::LogPType {
        use Residue4::*;
        let prob: na::SVector<Float, 4> = match &self {
            A => na::SVector::<Float, 4>::new(1.0, 0.0, 0.0, 0.0),
            C => na::SVector::<Float, 4>::new(0.0, 1.0, 0.0, 0.0),
            G => na::SVector::<Float, 4>::new(0.0, 0.0, 1.0, 0.0),
            T => na::SVector::<Float, 4>::new(0.0, 0.0, 0.0, 1.0),
            Residue4::None => na::SVector::<Float, 4>::new(0.25, 0.25, 0.25, 0.25),
            R => na::SVector::<Float, 4>::new(0.5, 0.0, 0.5, 0.0),
            Y => na::SVector::<Float, 4>::new(0.0, 0.5, 0.0, 0.5),
            S => na::SVector::<Float, 4>::new(0.0, 0.5, 0.5, 0.0),
            W => na::SVector::<Float, 4>::new(0.5, 0.0, 0.0, 0.5),
            K => na::SVector::<Float, 4>::new(0.0, 0.0, 0.5, 0.5),
            M => na::SVector::<Float, 4>::new(0.5, 0.5, 0.0, 0.0),
            B => na::SVector::<Float, 4>::new(0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            D => na::SVector::<Float, 4>::new(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0),
            H => na::SVector::<Float, 4>::new(1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0),
            V => na::SVector::<Float, 4>::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        };
        prob.map(Float::ln)
    }
}
impl ResidueTrait for Residue4 {
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>> {
        input.chars().map(Self::try_from)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Residue5 {
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
impl Residue5 {
    pub const U: Self = Self::T;
}

impl TryFrom<char> for Residue5 {
    type Error = FelsensteinError;
    fn try_from(char: char) -> Result<Self, Self::Error> {
        use Residue5::*;
        let res: Residue5 = match char.to_ascii_uppercase() {
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
            'U' => Residue5::U,
            '-' => Residue5::None,
            '.' => Residue5::None,
            'N' => Residue5::None,
            _ => return Err(FelsensteinError::INVALID_CHAR),
        };
        Ok(res)
    }
}
impl EntryTrait for Residue5 {
    const TOTAL: usize = 15;
    const DIM: usize = 5;
    const CHARS: usize = 1;
    type LogPType = na::SVector<Float, { Self::DIM }>;

    fn to_log_p(&self) -> Self::LogPType {
        use Residue5::*;
        let prob: na::SVector<Float, { Self::DIM }> = match &self {
            A => na::SVector::<Float, 5>::new(1.0, 0.0, 0.0, 0.0, 0.0),
            C => na::SVector::<Float, 5>::new(0.0, 1.0, 0.0, 0.0, 0.0),
            G => na::SVector::<Float, 5>::new(0.0, 0.0, 1.0, 0.0, 0.0),
            T => na::SVector::<Float, 5>::new(0.0, 0.0, 0.0, 1.0, 0.0),
            Residue5::None => na::SVector::<Float, 5>::new(0.0, 0.0, 0.0, 0.0, 1.0),
            R => na::SVector::<Float, 5>::new(0.5, 0.0, 0.5, 0.0, 0.0),
            Y => na::SVector::<Float, 5>::new(0.0, 0.5, 0.0, 0.5, 0.0),
            S => na::SVector::<Float, 5>::new(0.0, 0.5, 0.5, 0.0, 0.0),
            W => na::SVector::<Float, 5>::new(0.5, 0.0, 0.0, 0.5, 0.0),
            K => na::SVector::<Float, 5>::new(0.0, 0.0, 0.5, 0.5, 0.0),
            M => na::SVector::<Float, 5>::new(0.5, 0.5, 0.0, 0.0, 0.0),
            B => na::SVector::<Float, 5>::new(0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
            D => na::SVector::<Float, 5>::new(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
            H => na::SVector::<Float, 5>::new(1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 0.0),
            V => na::SVector::<Float, 5>::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0),
        };
        prob.map(Float::ln)
    }
}
impl ResidueTrait for Residue5 {
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>> {
        input.chars().map(Self::try_from)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ResiduePair<Res>(pub Res, pub Res);

impl<Residue, T, const RES_DIM: usize, const DIM: usize> EntryTrait for ResiduePair<Residue>
where
    Residue: EntryTrait<LogPType = na::SVector<T, RES_DIM>>,
    T: ScalarTrait,
    Const<RES_DIM>: DimMul<Const<RES_DIM>, Output = Const<DIM>>,
{
    const DIM: usize = Residue::DIM * Residue::DIM;
    const TOTAL: usize = Residue::TOTAL * Residue::TOTAL;
    const CHARS: usize = Residue::CHARS * 2;
    type LogPType = na::SVector<T, DIM>;

    /* TODO! this should be static */
    fn to_log_p(&self) -> Self::LogPType {
        let mut result = na::SVector::<T, DIM>::zeros();
        let (first, second) = (self.0, self.1);
        let log_p_first = first.to_log_p();
        let log_p_second = second.to_log_p();
        for a in 0..Residue::DIM {
            for b in 0..Residue::DIM {
                result[Residue::DIM * a + b] = log_p_first[a] + log_p_second[b];
            }
        }
        result
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
    pub const INVALID_CHAR: Self =
        Self::DeserializationError("Invalid character in residue sequence");
    /*
    pub fn invalid_residue(s: &str) -> Self {
        Self::DeserializationError(format!("Invalid residue:{}", s))
    }*/
}

pub type TwoTimesConst<const N: usize> = TwoTimes<Const<N>>;
pub type TwoTimes<N> = <N as DimAdd<N>>::Output;
pub type MatrixNNAllocated<T, N> =
    na::Matrix<T, N, N, <DefaultAllocator as Allocator<T, N, N>>::Buffer>;
pub trait Exponentiable: ToTypenum + DimName + DimMin<Self, Output = Self> {}
impl<T: ToTypenum + DimName + DimMin<Self, Output = Self>> Exponentiable for T {}

pub trait Doubleable
where
    Self: ToTypenum + DimAdd<Self>,
{
}
impl<T> Doubleable for T where T: ToTypenum + DimAdd<T> {}

/* pub trait ViableDim
where
    Self: ToTypenum + DimName + Doubleable,
    TwoTimes<Self>: DimName + Exponentiable,
{
}
impl<T> ViableDim for T
where
    T: ToTypenum + DimName + Doubleable,
    TwoTimes<T>: DimName + Exponentiable,
{
} */

pub trait ViableAllocator<T, const N: usize>
where
    Self: Allocator<T, TwoTimesConst<N>, TwoTimesConst<N>>
        + Allocator<T, Const<N>, Const<N>, Buffer = na::ArrayStorage<Float, N, N>>
        + Allocator<T, TwoTimesConst<N>>
        + Allocator<(usize, usize), TwoTimesConst<N>>,
    Const<N>: Doubleable,
{
}
impl<A, T, const N: usize> ViableAllocator<T, N> for A
where
    A: Allocator<T, TwoTimesConst<N>, TwoTimesConst<N>>
        + Allocator<(usize, usize), TwoTimesConst<N>>
        + Allocator<T, Const<N>, Const<N>, Buffer = na::ArrayStorage<Float, N, N>>
        + Allocator<T, TwoTimesConst<N>>,
    Const<N>: Doubleable,
{
}
