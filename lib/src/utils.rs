use ndarray::*;
use rand::prelude::*;
use std::f64;

pub type Vector = Array1<f64>;
pub type Matrix = Array2<f64>;

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

pub fn shuffle<T>(data: &mut [T]) {
    data.shuffle(&mut thread_rng());
}

pub fn imax(data: &Vector) -> usize {
    data.iter()
        .enumerate()
        .fold(
            (0, f64::MIN),
            |(i, max), (j, &val)| if val > max { (j, val) } else { (i, max) },
        )
        .0
}

pub fn as_matrix(data: &Vector) -> ArrayBase<ViewRepr<&f64>, Ix2> {
    ArrayView::from_shape((data.dim(), 1), data.view().into_slice().unwrap()).unwrap()
}
