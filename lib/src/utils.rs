use rand::prelude::*;
use rand_distr::StandardNormal;

pub mod matrix;
pub mod vector;

/*pub use matrix::Matrix;*/

pub fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}