use rand::prelude::*;
use rand_distr::StandardNormal;

mod matrix;
mod vector;

pub use matrix::*;
pub use vector::*;

pub fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}
