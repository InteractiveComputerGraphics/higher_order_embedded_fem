/// Poor man's approx assertion for matrices
#[macro_export]
macro_rules! assert_approx_matrix_eq {
    ($x:expr, $y:expr, abstol = $tol:expr) => {{
        let diff = $x - $y;

        let max_absdiff = diff.abs().max();
        let approx_eq = max_absdiff <= $tol;

        if !approx_eq {
            println!("abstol: {}", $tol);
            println!("left: {}", $x);
            println!("right: {}", $y);
            println!("diff: {:e}", diff);
        }
        assert!(approx_eq);
    }};
}

#[macro_export]
macro_rules! assert_panics {
    ($e:expr) => {{
        use std::panic::catch_unwind;
        use std::stringify;
        let expr_string = stringify!($e);
        let result = catch_unwind(|| $e);
        if result.is_ok() {
            panic!("assert_panics!({}) failed.", expr_string);
        }
    }};
}
