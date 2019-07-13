pub mod utils;
use utils::*;

pub type Data = (Vector<f64>, Vector<f64>);

pub struct Network {
    nlayers: isize,
    biases: Vector<Vector<f64>>,
    weights: Vector<Matrix<f64>>,
}

impl Network {
    pub fn new(layers: &Vector<isize>) -> Self {
        Self {
            nlayers: layers.shape(),
            biases: layers.map_range(1..0, &|&x| Vector::from_fn(x, &|_| randn())),
            weights: layers.zip_map_range(1..0, layers, 0..-1, &|(&x, &y)| Matrix::from_fn((x, y), &|_| randn())),
        }
    }

    pub fn train(
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        training_data: &mut Vector<Data>,
        test_data: Option<&Vector<Data>>,
    ) {
        for i in 0..epochs {
            training_data.shuffle();
            for j in (0..training_data.shape()).step_by(batch_size) {
                let batch = &training_data;
                self.train_batch(batch, learning_rate);
            }
            if let Some(test_data) = test_data {
                println!(
                    "Epoch {}: {} / {}",
                    i,
                    self.evaluate(test_data),
                    test_data.shape()
                );
            } else {
                println!("Epoch {} complete", i);
            }
        }
    }

    fn train_batch(&mut self, batch: &Vector<Data>, learning_rate: f64) {
        let (mut nabla_b, mut nabla_w) = self.nabla_zeros();
        for data in batch.iter() {
            let (delta_nabla_b, delta_nabla_w) = self.backpropagate(data);
            nabla_b.zip_apply(&delta_nabla_b, &|(&nb, dnb)| {nb += dnb});
            nabla_w.zip_apply(&delta_nabla_w, &|(&nw, dnw)| {nw += dnw});
        }
        self.biases.zip_apply(&nabla_b, &|(&b, nb)| {
            b -= &(&(learning_rate * nb) / batch.len() as f64)
        });
        self.weights.zip_apply(&nabla_w, &|(&w, nw)| {
            w -= &(&(learning_rate * nw) / batch.len() as f64)
        });
    }

    fn backpropagate(&self, (x, y): &Data) -> (Vector<Vector<f64>>, Vector<Matrix<f64>>) {
        let (mut nabla_b, mut nabla_w) = self.nabla_zeros();
        let mut activations = Vector::from_data(1, vec![x.data().clone()]);
        let mut zs: Vector<Vector<f64>> = Vector::empty();
        for (b, w) in self.biases.zip(&self.weights) {
            zs.push(&(w * activations[-1]) + b);
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

    fn nabla_zeros(&self) -> (Vector<Vector<f64>>, Vector<Matrix<f64>>) {
        (
            self.biases.map(&|b| Vector::from_val(b.shape(), 0.0)),
            self.weights.map(&|w| Matrix::from_val(w.shape(), 0.0)),
        )
    }

    fn evaluate(&self, test_data: &Vector<Data>) -> usize {
        test_data.sum_by(&|(x, y): &Data| (self.feedforward(x).argmax() == y.argmax()) as usize)
    }

    fn feedforward(&self, x: &Vector<f64>) -> Vector<f64> {
        let mut a = *x.clone();
        for (b, w) in self.biases.zip(&self.weights) {
            a = (&(w * &a) + b).map(&|&x| sigmoid(x));
        }
        a
    }

    pub fn data_from_vecs(x: Vec<f64>, y: Vec<f64>) -> Data {
        (Vector::from_data(x.len() as isize, x), Vector::from_data(y.len() as isize, y))
    }
}

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
    fn test_matrix_indexing() {
        let mut m = Matrix::from_val((5, 5), 0);
        m[(-2, 1)] = 4;
        assert_eq!(4, m[(3, -4)]);
    }
    
    #[test]
    fn test_add() {
        let x = Vector::from_data(4, vec![0.0, 1.0, 3.0, 2.0]);
        let y = Vector::from_data(4, vec![4.0, 7.0, 1.0, -5.0]);
        let mut z = &x + &y;
        assert_eq!(Vector::from_data(4, vec![4.0, 8.0, 4.0, -3.0]), z);
        z += &y;
        assert_eq!(Vector::from_data(4, vec![8.0, 15.0, 5.0, -8.0]), z);
    }

    #[test]
    fn test_mul() {
        let x = Matrix::from_data((2, 2), vec![0.0, 1.0, 3.0, 2.0]);
        let y = Matrix::from_data((2, 2), vec![4.0, 7.0, 1.0, -5.0]);
        assert_eq!(
            Matrix::from_data((2, 2), vec![21.0, 18.0, -15.0, -9.0]),
            &x * &y
        );
    }

    #[test]
    fn test_hadamard() {
        let x = Vector::from_data(4, vec![0.0, 1.0, 3.0, 2.0]);
        let y = Vector::from_data(4, vec![4.0, 7.0, 1.0, -5.0]);
        assert_eq!(
            Vector::from_data(4, vec![0.0, 7.0, 3.0, -10.0]),
            x.hadamard(&y)
        );
    }
}
