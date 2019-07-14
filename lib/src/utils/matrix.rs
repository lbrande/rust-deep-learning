use mkl::*;
use std::ops::*;

#[derive(Debug, PartialEq)]
pub struct Matrix {
    shape: (isize, isize),
    data: Vec<f64>,
}

impl Matrix {
    pub fn from_shape(shape: (isize, isize)) -> Self {
        assert!(shape.0 >= 0 && shape.1 >= 0);
        Self {
            shape,
            data: Vec::with_capacity((shape.0 * shape.1) as usize),
        }
    }

    pub fn from_fn(shape: (isize, isize), f: &Fn(isize) -> f64) -> Self {
        assert!(shape.0 >= 0 && shape.1 >= 0);
        Self {
            shape,
            data: (0..shape.0 * shape.1).map(f).collect(),
        }
    }

    pub fn from_vec(shape: (isize, isize), data: Vec<f64>) -> Self {
        assert!(shape.0 >= 0 && shape.1 >= 0);
        assert!((shape.0 * shape.1) as usize == data.len());
        Self {
            shape,
            data,
        }
    }

    pub fn shape(&self) -> (isize, isize) {
        self.shape
    }

    pub fn hadamard(&mut self, other: &Self) {
        assert!(self.shape == other.shape);
        unsafe {
            VDMUL_(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                other.data.as_ptr(),
                self.data.as_mut_ptr(),
            );
        }
    }

    pub fn imax(&self) -> isize {
        unsafe { IDAMAX(&(self.data.len() as i32), self.data.as_ptr(), &1) as isize }
    }

    pub fn apply(&mut self, f: &Fn(f64) -> f64) {
        self.data.iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn zip_apply(&mut self, other: &Self, f: &Fn((f64, f64)) -> f64) {
        self.data.iter_mut().zip(&other.data).for_each(|(x, &y)| *x = f((*x, y)));
    }

    pub fn set_ref_to(&mut self, other: &Self) {
        unsafe {
            *self.data.as_mut_ptr() = *other.data.as_ptr();
        }
    }

    pub fn dgemv( trans : char, a_matrix : &Self , x_vector : &Self, y_vector : &mut Self) {
        let m = a_matrix.shape.0 as i32;
        let n = a_matrix.shape.1 as i32;
        unsafe {
            DGEMV(
                &(trans as i8),
                &m,
                &n,
                &1.0,
                a_matrix.data.as_ptr(),
                &m,
                x_vector.data.as_ptr(),
                &1,
                &1.0,
                y_vector.data.as_mut_ptr(),
                &1,
            );
        }
    }

    pub fn dgemm( transa : char , transb : char, a_matrix : &Self , b_matrix : &Self, c_matrix : &mut Self) {
        let m = a_matrix.shape.0 as i32;
        let n = b_matrix.shape.1 as i32;
        let k = a_matrix.shape.1 as i32;
        let lda = if transa == 'N' { m } else { k };
        let ldb = if transa == 'N' { k } else { n };
        unsafe {
            DGEMM(
                &(transa as i8),
                &(transb as i8),
                &m,
                &n,
                &k,
                &1.0,
                a_matrix.data.as_ptr(),
                &lda,
                b_matrix.data.as_ptr(),
                &ldb,
                &1.0,
                c_matrix.data.as_mut_ptr(),
                &m,
            );
        }
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
}

impl Matrix {
    pub fn from_val(shape: (isize, isize), val: f64) -> Self {
        assert!(shape.0 >= 0 && shape.1 >= 0);
        Self {
            shape,
            data: vec![val; (shape.0 * shape.1) as usize],
        }
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let mut clone = Vec::with_capacity((self.shape.0 * self.shape.1) as usize);
        unsafe {
            DCOPY(&(self.data.len() as i32), self.data.as_ptr(), &1, clone.as_mut_ptr(), &1);
        }
        Self::from_vec(self.shape, clone)
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

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, other: &Matrix) {
        assert!(self.shape == other.shape);
        unsafe {
            VDADD_(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                other.data.as_ptr(),
                self.data.as_mut_ptr(),
            );
        }
    }
}

impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, other: &Matrix) {
        assert!(self.shape == other.shape);
        unsafe {
            VDSUB_(
                &(self.data.len() as i32),
                self.data.as_ptr(),
                other.data.as_ptr(),
                self.data.as_mut_ptr(),
            );
        }
    }
}

impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, other: f64) {
        unsafe {
            DSCAL(
                &(self.data.len() as i32),
                &other,
                self.data.as_mut_ptr(),
                &1,
            );
        }
    }
}