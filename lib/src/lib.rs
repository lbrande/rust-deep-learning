mod utils;

use utils::*;

pub struct Network {
    biases: Vec<Vector>,
    weights: Vec<Matrix>,
}

impl Network {
    pub fn new(layers: &[usize]) -> Self {
        let init = &layers[1..];
        let tail = &layers[..layers.len() - 1];
        Self {
            biases: map(&|&x| randn1(x), init),
            weights: zip_with(&|&x, &y| randn2(x, y), tail, init),
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

    fn train_batch(&mut self, batch: &[Data], learning_rate: f64) {
        let (mut nabla_b, mut nabla_w) = self.nabla_zeros();
        for data in batch {
            let (delta_nabla_b, delta_nabla_w) = self.backpropagate(data);
            nabla_b = zip_with(&|nb, dnb| nb + dnb, &nabla_b, &delta_nabla_b);
            nabla_w = zip_with(&|nw, dnw| nw + dnw, &nabla_w, &delta_nabla_w);
        }
        self.biases = zip_with(
            &|b, nb| b - learning_rate * nb / batch.len() as f64,
            &self.biases,
            &nabla_b,
        );
        self.weights = zip_with(
            &|w, nw| w - learning_rate * nw / batch.len() as f64,
            &self.weights,
            &nabla_w,
        );
    }

    fn backpropagate(&self, data: &Data) -> (Vec<Vector>, Vec<Matrix>) {
        let (mut nabla_b, mut nabla_w) = self.nabla_zeros();
        (nabla_b, nabla_w)
    }

    fn nabla_zeros(&self) -> (Vec<Vector>, Vec<Matrix>) {
        (
            map(&|b: &Vector| zeros1(b.len()), &self.biases),
            map(&|w: &Matrix| zeros2(w.nrows(), w.ncols()), &self.weights),
        )
    }

    fn evaluate(&self, test_data: &[Data]) -> usize {
        sum_by(
            &|(x, y): &Data| (self.feedforward(x).imax() == y.imax()) as usize,
            test_data,
        )
    }

    fn feedforward(&self, input: &Vector) -> Vector {
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
