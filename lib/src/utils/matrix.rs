use mkl::blas::*;
use mkl::vml::*;
use std::ops::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    shape: (isize, isize),
    data: Vec<f64>,
    transposed: bool,
}

impl Matrix {
    pub fn from_fn(shape: (isize, isize), f: &Fn(isize) -> f64) -> Self {
        assert!(shape.0 >= 0 && shape.1 >= 0);
        Self {
            shape,
            data: (0..shape.0 * shape.1).map(f).collect(),
            transposed: false,
        }
    }

    pub fn from_vec(shape: (isize, isize), data: Vec<f64>) -> Self {
        assert!(shape.0 >= 0 && shape.1 >= 0);
        assert!((shape.0 * shape.1) as usize == data.len());
        Self {
            shape,
            data,
            transposed: false,
        }
    }

    pub fn shape(&self) -> (isize, isize) {
        self.shape
    }

    pub fn hadamard(&self, other: &Self) -> Self {
        assert!(self.shape == other.shape);
        let mut result = Matrix::from_val(self.shape, 0.0);
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

    pub fn transpose(&mut self) -> &mut Self {
        self.transposed = !self.transposed;
        self
    }

    fn fix_index(&self, mut index: (isize, isize)) -> usize {
        assert!(index.0 < self.shape.0);
        if index.0 < 0 {
            index.0 += self.shape.0;
        }
        if index.1 < 0 {
            index.1 += self.shape.1;
        }
        index.1 as usize * self.shape.0 as usize + index.0 as usize
    }

    fn transpose_char(&self) -> i8 {
        if self.transposed {
            'T' as i8
        } else {
            'N' as i8
        }
    }
}

impl Matrix {
    pub fn from_val(shape: (isize, isize), val: f64) -> Self {
        assert!(shape.0 >= 0 && shape.1 >= 0);
        Self {
            shape,
            data: vec![val; (shape.0 * shape.1) as usize],
            transposed: false,
        }
    }
}

impl Index<(isize, isize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (isize, isize)) -> &Self::Output {
        &self.data[self.fix_index(index)]
    }
}

impl IndexMut<(isize, isize)> for Matrix {
    fn index_mut(&mut self, index: (isize, isize)) -> &mut Self::Output {
        let index = self.fix_index(index);
        &mut self.data[index]
    }
}

impl Add<Self> for &Matrix {
    type Output = Matrix;

    fn add(self, other: Self) -> Self::Output {
        assert!(self.shape == other.shape);
        let mut result = Matrix::from_val(self.shape, 0.0);
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

impl Sub<Self> for &Matrix {
    type Output = Matrix;

    fn sub(self, other: Self) -> Self::Output {
        assert!(self.shape == other.shape);
        let mut result = Matrix::from_val(self.shape, 0.0);
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

impl Mul<Self> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: Self) -> Self::Output {
        assert!(self.shape.1 == other.shape.0);
        let mut result = Matrix::from_val((self.shape.0, other.shape.1), 0.0);
        let m = self.shape.0 as i32;
        let n = other.shape.1 as i32;
        let k = self.shape.1 as i32;
        let lda = if self.transposed { k } else { m };
        let ldb = if other.transposed { n } else { k };
        unsafe {
            DGEMM(
                &self.transpose_char(),
                &other.transpose_char(),
                &m,
                &n,
                &k,
                &1.0,
                self.data.as_ptr(),
                &lda,
                other.data.as_ptr(),
                &ldb,
                &1.0,
                result.data.as_mut_ptr(),
                &m,
            );
        }
        result
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: f64) -> Self::Output {
        let mut result = Matrix::from_val(self.shape, 0.0);
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

impl Mul<&Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Self::Output {
        let mut result = Matrix::from_val(other.shape, 0.0);
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

impl Div<f64> for &Matrix {
    type Output = Matrix;

    fn div(self, other: f64) -> Self::Output {
        self.mul(1.0 / other)
    }
}

impl Div<&Matrix> for f64 {
    type Output = Matrix;

    fn div(self, other: &Matrix) -> Self::Output {
        (1.0 / self).mul(other)
    }
}