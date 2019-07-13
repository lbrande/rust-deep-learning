use rand::prelude::*;
use rand::seq::SliceRandom;
use std::isize;
use std::iter::Sum;
use std::ops::*;

pub trait VectorUtils<'a, T: 'a> {
    fn data(&'a self) -> &'a [T];

    fn len(&'a self) -> isize {
        self.data().len() as isize
    }

    fn is_empty(&'a self) -> bool {
        self.data().is_empty()
    }

    fn map<U>(&'a self, f: &Fn(&T) -> U) -> Vector<U> {
        Vector::from_vec(self.data().iter().map(f).collect())
    }

    fn zip_map<U: 'a, V, W: VectorUtils<'a, U>>(&'a self, other: &'a W, f: &Fn((&T, &U)) -> V) -> Vector<V> {
        Vector::from_vec(self.data().iter().zip(other.data()).map(f).collect())
    }

    fn map_sum<U: Sum>(&'a self, f: &Fn(&T) -> U) -> U {
        self.data().iter().map(f).sum()
    }

    fn fix_start(&'a self, start: isize) -> usize {
        if start < 0 {
            (start + self.len()) as usize
        } else {
            start as usize
        }
    }

    fn fix_end(&'a self, end: isize) -> usize {
        if end <= 0 {
            (end + self.len()) as usize
        } else {
            end as usize
        }
    }

    fn fix_range(&'a self, range: Range<isize>) -> Range<usize> {
        self.fix_start(range.start)..self.fix_end(range.end)
    }
}

pub trait VectorUtilsMut<'a, T: 'a> {
    fn data_mut(&'a mut self) -> &'a mut [T];

    fn apply<U>(&'a mut self, f: &Fn(&mut T)) {
        self.data_mut().iter_mut().for_each(f);
    }

    fn zip_apply<U: 'a, V: VectorUtilsMut<'a, U>>(&'a mut self, other: &'a mut V, f: &Fn((&mut T, &mut U))) {
        self.data_mut().iter_mut().zip(other.data_mut()).for_each(f);
    }
}

pub trait VectorUtilsCopy<'a, T: 'a>: VectorUtilsMut<'a, T> {
    fn shuffle(&'a mut self) {
        self.data_mut().shuffle(&mut thread_rng());
    }
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T> Vector<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }
    
    pub fn from_fn(len: isize, f: &Fn(isize) -> T) -> Self {
        assert!(len >= 0);
        Self {
            data: (0..len).map(f).collect(),
        }
    }

    pub fn from_vec(data: Vec<T>) -> Self {
        assert!(data.len() <= isize::MAX as usize);
        Self {
            data,
        }
    }
    
    pub fn push(&mut self, val: T) {
        self.data.push(val);
    }
}

impl<T: Copy> Vector<T> {
    pub fn from_val(len: isize, val: T) -> Self {
        assert!(len >= 0);
        Self { data: vec![val; len as usize] }
    }
}

impl<T> Index<isize> for Vector<T> {
    type Output = T;

    fn index(&self, index: isize) -> &Self::Output {
        &self.data[self.fix_start(index)]
    }
}

impl<T> IndexMut<isize> for Vector<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let index = self.fix_start(index);
        &mut self.data[index]
    }
}

impl<T> Index<Range<isize>> for Vector<T> {
    type Output = [T];

    fn index(&self, range: Range<isize>) -> &Self::Output {
        &self.data[self.fix_range(range)]
    }
}

impl<T> IndexMut<Range<isize>> for Vector<T> {
    fn index_mut(&mut self, range: Range<isize>) -> &mut Self::Output {
        let range = self.fix_range(range);
        &mut self.data[range]
    }
}

impl<T> Index<RangeFrom<isize>> for Vector<T> {
    type Output = [T];

    fn index(&self, range: RangeFrom<isize>) -> &Self::Output {
        &self.data[self.fix_start(range.start)..]
    }
}

impl<T> IndexMut<RangeFrom<isize>> for Vector<T> {
    fn index_mut(&mut self, range: RangeFrom<isize>) -> &mut Self::Output {
        let start = self.fix_start(range.start);
        &mut self.data[start..]
    }
}

impl<T> Index<RangeTo<isize>> for Vector<T> {
    type Output = [T];

    fn index(&self, range: RangeTo<isize>) -> &Self::Output {
        &self.data[..self.fix_end(range.end)]
    }
}

impl<T> IndexMut<RangeTo<isize>> for Vector<T> {
    fn index_mut(&mut self, range: RangeTo<isize>) -> &mut Self::Output {
        let end = self.fix_end(range.end);
        &mut self.data[..end]
    }
}

impl<T> Index<RangeFull> for Vector<T> {
    type Output = [T];

    fn index(&self, _: RangeFull) -> &Self::Output {
        &self.data[..]
    }
}

impl<T> IndexMut<RangeFull> for Vector<T> {
    fn index_mut(&mut self, _: RangeFull) -> &mut Self::Output {
        &mut self.data[..]
    }
}

impl<'a, T: 'a> VectorUtils<'a, T> for Vector<T> {
    fn data(&'a self) -> &'a [T] {
        &self.data
    }
}

impl<'a, T: 'a> VectorUtilsMut<'a, T> for Vector<T> {
    fn data_mut(&'a mut self) -> &'a mut [T] {
        &mut self.data
    }
}

impl<'a, T: 'a> VectorUtilsCopy<'a, T> for Vector<T> {}

impl<'a, T: 'a> VectorUtils<'a, T> for &'a [T] {
    fn data(&'a self) -> &'a [T] {
        self
    }
}

impl<'a, T: 'a> VectorUtilsMut<'a, T> for &'a mut [T] {
    fn data_mut(&'a mut self) -> &'a mut [T] {
        self
    }
}

impl<'a, T: 'a> VectorUtilsCopy<'a, T> for &'a mut [T] {}