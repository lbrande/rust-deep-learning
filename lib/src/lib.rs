mod utils;

use nalgebra::*;
use utils::*;

type Data = (DVector<f64>, DVector<f64>);

pub struct Network {
    biases: Vec<DVector<f64>>,
    weights: Vec<DMatrix<f64>>,
}

impl Network {
    pub fn new(layers: &[usize]) -> Self {
        let init = &layers[1..];
        let tail = &layers[..layers.len() - 1];
        Self {
            biases: map(&|&x| randn_vector(x), init),
            weights: zip_with(&|&x, &y| randn_matrix(x, y), tail, init),
        }
    }

    pub fn train(
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        training_data: &mut [Data],
        test_data: Option<&[Data]>,
    ) {
        for i in 0..epochs {
            shuffle(training_data);
            for j in (0..training_data.len()).step_by(batch_size) {
                let batch = &training_data[j..j + batch_size];
                self.train_batch(batch, learning_rate);
            }
            if let Some(test_data) = test_data {
                println!(
                    "Epoch {}: {} / {}",
                    i,
                    self.evaluate(test_data),
                    test_data.len()
                );
            } else {
                println!("Epoch {} complete", i);
            }
        }
    }

    fn train_batch(&mut self, batch: &[Data], learning_rate: f64) {}

    fn evaluate(&self, test_data: &[Data]) -> usize {
        sum_by(
            &|(x, y): &Data| (self.feedforward(x).imax() == y.imax()) as usize,
            test_data,
        )
    }

    fn feedforward(&self, input: &DVector<f64>) -> DVector<f64> {
        let (b, w) = (&self.biases[0], &self.weights[0]);
        let mut a = w * input + b;
        for (b, w) in zip(&self.biases[1..], &self.weights[1..]) {
            a = w * a + b
        }
        a
    }
}

#[cfg(test)]
mod tests {}
