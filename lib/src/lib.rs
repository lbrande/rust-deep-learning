use nalgebra as na;
use rand::prelude::*;
use rand_distr::StandardNormal;

type Vector = na::DVector<f64>;
type Matrix = na::DMatrix<f64>;

fn map<T: Copy, U>(f: &Fn(T) -> U, slice: &[T]) -> Vec<U> {
    slice.iter().map(|&t| f(t)).collect()
}

fn zip_with<T: Copy, U>(f: &Fn(T, T) -> U, slice_0: &[T], slice_1: &[T]) -> Vec<U> {
    slice_0
        .iter()
        .zip(slice_1)
        .map(|(&t_0, &t_1)| f(t_0, t_1))
        .collect()
}

fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

fn randn_vector(rows: usize) -> Vector {
    Vector::from_vec((0..rows).map(|_| randn()).collect())
}

fn randn_matrix(rows: usize, columns: usize) -> Matrix {
    Matrix::from_vec(rows, columns, (0..rows*columns).map(|_| randn()).collect())
}

pub struct Network {
    biases: Vec<Vector>,
    weights: Vec<Matrix>,
}

impl Network {
    pub fn new(layers: &[usize]) -> Self {
        let init = &layers[1..];
        let tail = &layers[..layers.len() - 1];
        Self {
            biases: map(&randn_vector, init),
            weights: zip_with(&randn_matrix, tail, init),
        }
    }
}

#[cfg(test)]
mod tests {}
