mod utils;
use ndarray::*;
use ndarray_rand::RandomExt;
use rand::distributions::StandardNormal;
use utils::*;

pub struct Network {
    nlayers: usize,
    biases: Vec<Array1<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    fn new(layers: &[usize]) -> Self {
        Self {
            nlayers: layers.len(),
            biases: layers[1..]
                .iter()
                .map(|&x| Array1::random(x, StandardNormal))
                .collect(),
            weights: layers[1..]
                .iter()
                .zip(&layers[..layers.len() - 1])
                .map(|(&x, &y)| Array2::random((x, y), StandardNormal))
                .collect(),
        }
    }
}
