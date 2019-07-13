use crate::utils::*;
use rand::seq::SliceRandom;

#[derive(Clone, Debug, PartialEq)]
pub struct Vector<T> {
    shape: isize,
    data: Vec<T>,
}

impl<T> Vector<T> {
    pub fn from_fn(shape: isize, f: &Fn(isize) -> T) -> Self {
        Self::illegal_shape_panic(shape);
        Self {
            shape,
            data: (0..shape).map(f).collect(),
        }
    }

    pub fn from_data(shape: isize, data: Vec<T>) -> Self {
        Self::missmatched_shape_panic(shape, &data);
        Self {
            shape: data.len() as isize,
            data,
        }
    }

    pub fn empty() -> Self {
        Self {
            shape: 0,
            data: Vec::new(),
        }
    }

    pub fn shape(&self) -> isize {
        self.shape
    }

    pub fn data(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    pub fn apply<U>(&self, f: &Fn(&mut T)) {
        self.data.iter_mut().for_each(f);
    }

    pub fn map<U>(&self, f: &Fn(&T) -> U) -> Vector<U> {
        Vector::from_data(self.shape, self.data.iter().map(f).collect())
    }

    pub fn map_range<U>(&self, range: Range<isize>, f: &Fn(&T) -> U) -> Vector<U> {
        Vector::from_data(
            self.shape,
            self.data[self.fix_range(range)].iter().map(f).collect(),
        )
    }

    pub fn zip_map<U, V>(&self, other: &Vector<U>, f: &Fn((&T, &U)) -> V) -> Vector<V> {
        Vector::from_data(
            self.shape,
            self.zip(other).map(f).collect(),
        )
    }

    pub fn zip_map_range<U, V>(
        &self,
        self_range: Range<isize>,
        other: &Vector<U>,
        other_range: Range<isize>,
        f: &Fn((&T, &U)) -> V,
    ) -> Vector<V> {
        Vector::from_data(
            self.shape,
            self.data[self.fix_range(self_range)]
                .iter()
                .zip(&other.data[other.fix_range(other_range)])
                .map(f)
                .collect(),
        )
    }

    pub fn zip_apply<U>(&self, other: &Vector<U>, f: &Fn((&mut T, &mut U))) {
        self.data.iter_mut().zip(&mut other.data).for_each(f);
    }

    pub fn zip<'a, U>(&'a self, other: &'a Vector<U>) -> Zip<Iter<'a, T>, Iter<'a, U>> {
        self.data.iter().zip(&other.data)
    }

    pub fn sum_by<U: Sum>(&self, f: &Fn(&T) -> U) -> U {
        self.iter().map(f).sum()
    }

    pub fn push(&mut self, val: T) {
        self.data.push(val);
        self.shape += 1;
    }

    fn fix_index(&self, start: isize) -> usize {
        if start < 0 {
            (start + self.shape) as usize
        } else {
            start as usize
        }
    }

    fn fix_range(&self, range: Range<isize>) -> Range<usize> {
        self.fix_index(range.start)..self.fix_end(range.end)
    }

    fn fix_end(&self, end: isize) -> usize {
        if end <= 0 {
            (end + self.shape) as usize
        } else {
            end as usize
        }
    }

    fn illegal_shape_panic(shape: isize) {
        if shape < 0 {
            panic!(ILLEGAL_SHAPE);
        }
    }

    fn missmatched_shape_panic(shape: isize, data: &[T]) {
        if shape as usize != data.len() {
            panic!(MISSMATCHED_SHAPE);
        }
    }
    fn missmatched_shapes_mul_mat_panic(other: &Matrix<T>) {
        if 1 != other.shape().0 {
            panic!(MISSMATCHED_SHAPES);
        }
    }

    fn missmatched_shapes_panic(&self, other: &Self) {
        if self.shape() != other.shape() {
            panic!(MISSMATCHED_SHAPES);
        }
    }
}

impl<T: Copy> Vector<T> {
    pub fn from_val(shape: isize, val: T) -> Self {
        Self::illegal_shape_panic(shape);
        Self {
            shape,
            data: vec![val; shape as usize],
        }
    }

    pub fn shuffle(&mut self) {
        self.data.shuffle(&mut thread_rng());
    }
}

impl<T> Index<isize> for Vector<T> {
    type Output = T;

    fn index(&self, index: isize) -> &Self::Output {
        &self.data[self.fix_index(index)]
    }
}
impl<T> IndexMut<isize> for Vector<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let index = self.fix_index(index);
        &mut self.data[index]
    }
}

impl<T> Deref for Vector<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.data.deref()
    }
}

impl Add<Self> for &Vector<f64> {
    type Output = Vector<f64>;

    fn add(self, other: Self) -> Self::Output {
        self.missmatched_shapes_panic(other);
        let mut result = Vector::from_val(self.shape, 0.0);
        unsafe {
            VDADD_(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                other.data.as_ptr(),
                result.data.as_mut_ptr(),
            );
        }
        result
    }
}

impl Sub<Self> for &Vector<f64> {
    type Output = Vector<f64>;

    fn sub(self, other: Self) -> Self::Output {
        self.missmatched_shapes_panic(other);
        let mut result = Vector::from_val(self.shape, 0.0);
        unsafe {
            VDSUB_(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                other.data.as_ptr(),
                result.data.as_mut_ptr(),
            );
        }
        result
    }
}

impl Mul<&mut Matrix<f64>> for &Vector<f64> {
    type Output = Matrix<f64>;

    fn mul(self, other: &mut Matrix<f64>) -> Self::Output {
        Vector::missmatched_shapes_mul_mat_panic(other);
        let mut result = Matrix::from_val((self.shape, other.shape().1), 0.0);
        let m = self.shape as i32;
        let n = other.shape().1 as i32;
        let k = 1;
        let lda = m;
        let ldb = if other.transposed() { n } else { k };
        unsafe {
            DGEMM(
                &('T' as i8),
                &other.get_transpose_char(),
                &m,
                &n,
                &k,
                &1.0,
                self.data.as_ptr(),
                &lda,
                other.data().as_ptr(),
                &ldb,
                &1.0,
                result.data().as_mut_ptr(),
                &m,
            );
        }
        result
    }
}

impl Mul<f64> for &Vector<f64> {
    type Output = Vector<f64>;

    fn mul(self, other: f64) -> Self::Output {
        let mut result = Vector::from_val(self.shape, 0.0);
        unsafe {
            DCOPY(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                &1,
                result.data.as_mut_ptr(),
                &1
            );
            DSCAL(
                &(self.data.len() as i32),
                &other,
                result.data.as_mut_ptr(),
                &1,
            );
        }
        result
    }
}

impl Mul<&Vector<f64>> for f64 {
    type Output = Vector<f64>;

    fn mul(self, other: &Vector<f64>) -> Self::Output {
        let mut result = Vector::from_val(other.shape, 0.0);
        unsafe {
            DCOPY(
                &(other.data.len() as i32),
                other.data.as_ptr(),
                &1,
                result.data.as_mut_ptr(),
                &1
            );
            DSCAL(
                &(other.data.len() as i32),
                &self,
                result.data.as_mut_ptr(),
                &1,
            );
        }
        result
    }
}

impl Div<f64> for &Vector<f64> {
    type Output = Vector<f64>;

    fn div(self, other: f64) -> Self::Output {
        self.mul(1.0 / other)
    }
}

impl Div<&Vector<f64>> for f64 {
    type Output = Vector<f64>;

    fn div(self, other: &Vector<f64>) -> Self::Output {
        (1.0 / self).mul(other)
    }
}

impl Vector<f64> {
    pub fn hadamard(&self, other: &Self) -> Self {
        self.missmatched_shapes_panic(other);
        let mut result = Vector::from_val(self.shape, 0.0);
        unsafe {
            VDMUL_(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                other.data.as_ptr(),
                result.data.as_mut_ptr(),
            );
        }
        result
    }

    pub fn argmax(&self) -> isize {
        unsafe {
            IDAMAX(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                &1,
            ) as isize
        }
    }
}
