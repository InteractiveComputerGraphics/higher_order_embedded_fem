#ifndef LP_BFP_H
#define LP_BFP_H

extern "C" {
    /// Attempts to solve the given Linear Program, storing the result in `x`.
    ///
    /// The LP is given by
    ///  min c^T x
    ///  s.t. Ax = b
    ///      lb <= x <= ub
    /// Let A be an m x n matrix. Then the LP has m equality constraints and
    /// n (possibly bounded) variables.
    ///
    /// \param x Output array of length n.
    /// \param c Array of length n.
    /// \param A Row-major m x n matrix stored as array of length mxn.
    /// \param b Array of length m.
    /// \param lb Array of length n.
    /// \param ub Array of length n.
    /// \param num_constraints m.
    /// \param num_variables n.
    /// \param verbose Whether or not to print debug output to stdout.
    /// \return An error code. 0 denotes success.
    int lp_bfp_solve_lp(double * x,
                        const double * c,
                        const double * A,
                        const double * b,
                        const double * lb,
                        const double * ub,
                        int num_constraints,
                        int num_variables,
                        bool verbose);
};

#endif // LP_BFP_H
