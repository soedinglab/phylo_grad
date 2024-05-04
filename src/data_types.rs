/* TODO: soft mapping for pairs */

use crate::ndarray::prelude::*;
use crate::num_enum::{IntoPrimitive, TryFromPrimitive};

#[derive(Debug, Copy, Clone, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Residue {
    A,
    C,
    G,
    T,
    None,
}

impl Residue {
    pub const TOTAL: usize = 5;
    pub const DIM: usize = 5;
}
#[derive(Debug, Copy, Clone, TryFromPrimitive, IntoPrimitive)]
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
    pub const TOTAL: usize = 15;
    pub const DIM: usize = 4;
    pub const U: Self = Self::T;

    pub fn to_log_p(&self) -> [Float; ResidueExtended::DIM] {
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
}

/* TODO this should be a try_from that fails in the default case */
impl From<char> for ResidueExtended {
    fn from(char: char) -> Self {
        use ResidueExtended::*;
        match char.to_ascii_uppercase() {
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
            //'-' => ResidueExtended::None,
            //'.' => ResidueExtended::None,
            //'N' => ResidueExtended::None,
            _ => ResidueExtended::None,
        }
    }
}

/* TODO #[repr(u8)] */
#[derive(Debug)]
struct ResiduePair(Residue, Residue);

#[derive(Debug)]
pub enum FelsensteinError {
    LogicError(&'static str),
}

impl std::fmt::Display for FelsensteinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self::LogicError(str) = self;
        write!(f, "{}", str)
    }
}
impl std::error::Error for FelsensteinError {}

pub type Entry = ResidueExtended;
/* TODO: f32 not allowed by expm crate. That crate is only used for debugging purposes though. */
pub type Float = f64;
pub type Id = usize;
pub type LogPType = [Float; Entry::DIM];
pub type RateType = na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>;
