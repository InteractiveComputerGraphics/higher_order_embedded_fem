use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

use fenris::nalgebra::allocator::Allocator;
use fenris::nalgebra::{DefaultAllocator, DimName, Point, RealField, VectorN};
use itertools::izip;

pub trait IfTrue {
    fn if_true<T>(&self, then_some: T) -> Option<T>;
}

impl IfTrue for bool {
    /// Maps `true` to a `Some(then_some)` or `false` to `None`.
    fn if_true<T>(&self, then_some: T) -> Option<T> {
        if *self {
            Some(then_some)
        } else {
            None
        }
    }
}

// TODO: Move this somewhere
pub fn apply_displacements<T, D>(x: &mut [Point<T, D>], x0: &[Point<T, D>], displacements: &[VectorN<T, D>])
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    assert_eq!(x.len(), x0.len());
    assert_eq!(x0.len(), displacements.len());
    for (v, v0, d) in izip!(x, x0, displacements) {
        *v = v0 + d;
    }
}

/// Takes an iterator of indices that yields unsorted, possibly duplicate indices
/// and maps the indices to a new set of indices [0, N), where `N` is the number of
/// unique original indices.
///
/// Returns a tuple consisting of the number of indices in the new index set and
/// a mapping from old to new indices.
pub fn relabel_indices(original_indices: impl IntoIterator<Item = usize>) -> (usize, HashMap<usize, usize>) {
    let iter = original_indices.into_iter();
    let ordered_indices: BTreeSet<_> = iter.collect();
    let num_new_indices = ordered_indices.len();
    let mapping = ordered_indices
        .into_iter()
        .enumerate()
        .map(|(new_idx, old_idx)| (old_idx, new_idx))
        .collect();

    (num_new_indices, mapping)
}

pub fn difference<T: Eq + Hash + Clone>(a: impl Iterator<Item = T>, b: impl Iterator<Item = T>) -> Vec<T> {
    let set_a: HashSet<_> = a.collect();
    let set_b: HashSet<_> = b.collect();
    set_a.difference(&set_b).map(|v| (*v).clone()).collect()
}

pub fn intersection<T: Eq + Hash + Clone>(a: impl Iterator<Item = T>, b: impl Iterator<Item = T>) -> Vec<T> {
    let set_a: HashSet<_> = a.collect();
    let set_b: HashSet<_> = b.collect();
    set_a.intersection(&set_b).map(|v| (*v).clone()).collect()
}

pub fn all_items_unique<T: Eq + Hash>(iter: impl IntoIterator<Item = T>) -> bool {
    let mut uniq = HashSet::new();
    iter.into_iter().all(move |x| uniq.insert(x))
}
