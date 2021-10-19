use std::any::TypeId;
use std::mem::transmute;

pub fn is_same_type<T, U>() -> bool
where
    T: 'static,
    U: 'static,
{
    TypeId::of::<T>() == TypeId::of::<U>()
}

pub fn transmute_identical_slice<T, U>(slice: &[T]) -> Option<&[U]>
where
    T: 'static,
    U: 'static,
{
    if is_same_type::<T, U>() {
        Some(unsafe { transmute(slice) })
    } else {
        None
    }
}

pub fn transmute_identical_slice_mut<T, U>(slice: &mut [T]) -> Option<&mut [U]>
    where
        T: 'static,
        U: 'static,
{
    if is_same_type::<T, U>() {
        Some(unsafe { transmute(slice) })
    } else {
        None
    }
}
