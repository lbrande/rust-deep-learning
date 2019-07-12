pub mod utils;
use utils::*;

/*pub type Data = (Vector, Vector);

pub struct Network {
    nlayers: usize,
    biases: Vec<Vector>,
    weights: Vec<Matrix>,
}

impl Network {
    pub fn new(layers: &[usize]) -> Self {
        let init = layers.slice(0, -1);
        let tail = layers.slice(1, 0);
        Self {
            nlayers: layers.len(),
            biases: tail.map(&|&x| randn1(x)),
            weights: tail.zip_map(init, &|&x, &y| randn2(x, y)),
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
            training_data.shuffle();
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
            nabla_b = nabla_b.zip_map(&delta_nabla_b, &|nb, dnb| nb + dnb);
            nabla_w = nabla_w.zip_map(&delta_nabla_w, &|nw, dnw| nw + dnw);
        }
        self.biases = self.biases.zip_map(&nabla_b, &|b, nb| {
            b - learning_rate * nb / batch.len() as f64
        });
        self.weights = self.weights.zip_map(&nabla_w, &|w, nw| {
            w - learning_rate * nw / batch.len() as f64
        });
    }

    fn backpropagate(&self, (x, y): &Data) -> (Vec<Vector>, Vec<Matrix>) {
        let (mut nabla_b, mut nabla_w) = self.nabla_zeros();
        let mut activations = vec![x.clone()];
        let mut zs = Vec::new();
        for (b, w) in self.biases.zip(&self.weights) {
            zs.push(w * activations.elem(-1) + b);
            activations.push(zs.elem(-1).sigmoid());
        }
        let mut delta = (activations.elem(-1) - y).hadamard(&(zs.elem(-1).sigmoid_prime()));
        nabla_b.set_elem(-1, delta.clone());
        nabla_w.set_elem(-1, delta.clone() * activations.elem(-2).transpose());
        for i in 2..self.nlayers {
            let i = i as i32;
            let z = zs.elem(-i);
            let sp = z.sigmoid_prime();
            delta = (self.weights.elem(-i + 1).transpose() * delta).hadamard(&sp);
            nabla_b.set_elem(-i, delta.clone());
            nabla_w.set_elem(-i, delta.clone() * activations.elem(-i - 1).transpose());
        }
        (nabla_b, nabla_w)
    }

    fn nabla_zeros(&self) -> (Vec<Vector>, Vec<Matrix>) {
        (
            self.biases.map(&|b: &Vector| zeros1(b.len())),
            self.weights.map(&|w: &Matrix| zeros2(w.nrows(), w.ncols())),
        )
    }

    fn evaluate(&self, test_data: &[Data]) -> usize {
        test_data.sum_by(&|(x, y): &Data| (self.feedforward(x).imax() == y.imax()) as usize)
    }

    fn feedforward(&self, x: &Vector) -> Vector {
        let mut a = x.clone();
        for (b, w) in self.biases.zip(&self.weights) {
            a = (w * a + b).sigmoid();
        }
        a
    }

    pub fn data_from_vecs(x: Vec<f64>, y: Vec<f64>) -> Data {
        (from_vec1(x), from_vec1(y))
    }
}*/

#[cfg(test)]
mod tests {
    use crate::utils::*;

    #[test]
    fn test_indexing() {
        let mut v = Vector::from_val(5, 0);
        v[-2] = 4;
        assert_eq!(4, v[3]);
    }

    #[test]
    fn test_slicing() {
        let mut v = Vector::from_val(5, 0);
        let s = &mut v[-2..0];
        s[0] = 4;
        assert_eq!(4, v[3]);
        assert_eq!(&[4, 0], &v[3..]);
        assert_eq!(&[0, 0, 0, 4], &v[..-1]);
        assert_eq!(&[0, 0, 0, 4, 0], &v[..]);
    }

    #[test]
    fn test_matrix_indexing() {
        let mut m = Matrix::from_val(5, 5, 0);
        m[(-2, 1)] = 4;
        assert_eq!(4, m[(3, -4)]);
    }
    #[test]
    fn test_add() {
        let x = Vector::from_data(vec![0.0, 1.0, 3.0, 2.0]);
        let y =  Vector::from_data(vec![4.0, 7.0, 1.0, -5.0]);
        assert_eq!(Vector::from_data(vec![4.0, 8.0, 4.0, -3.0]), &x + &y);
    }
}
