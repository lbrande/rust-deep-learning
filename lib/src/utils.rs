use mkl::blas::*;
use mkl::vml::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

use std::iter::*;
use std::slice::Iter;
use std::ops::*;

mod matrix;
mod vector;

pub use matrix::Matrix;
pub use vector::Vector;

static ILLEGAL_SHAPE: &'static str = "illegal shape";
static MISSMATCHED_SHAPE: &'static str = "missmatched shape";
static MISSMATCHED_SHAPES: &'static str = "missmatched shapes";
static INDEX_OUT_OF_BOUNDS: &'static str = "index out of bounds";

pub fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}
