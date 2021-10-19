#include "lp-bfp.h"

#include <ortools/lp_data/lp_data.h>
#include <ortools/glop/lp_solver.h>
#include <iostream>

int lp_bfp_solve_lp(double * x,
                    const double * c,
                    const double * A,
                    const double * b,
                    const double * lb,
                    const double * ub,
                    int num_constraints,
                    int num_variables,
                    bool verbose)
{
    using operations_research::glop::LinearProgram;
    using operations_research::glop::ColIndex;
    using operations_research::glop::ProblemStatus;
    using operations_research::glop::LPSolver;
    using operations_research::glop::GlopParameters;
    using operations_research::glop::GlopParameters_SolverBehavior;
    using std::cout;

    LinearProgram lp;

    std::vector<ColIndex> variables;
    for (int i = 0; i < num_variables; ++i)
    {
        const auto variable = lp.CreateNewVariable();
        lp.SetVariableBounds(variable, lb[i], ub[i]);
        lp.SetObjectiveCoefficient(variable, c[i]);
        variables.push_back(variable);
    }

    for (int i = 0; i < num_constraints; ++i)
    {
        const auto constraint_row = lp.CreateNewConstraint();
        lp.SetConstraintBounds(constraint_row, b[i], b[i]);

        for (int j = 0; j < num_variables; ++j)
        {
            const auto linear_matrix_index = num_variables * i + j;
            const auto a_ij = A[linear_matrix_index];
            // GLOP uses sparse matrices under the hood, and it does not like explicit zeros.
            if (a_ij != 0.0) {
                lp.SetCoefficient(constraint_row, variables[j], a_ij);
            }
        }
    }

    if (verbose) {
        cout << "Constructed LP with " << num_variables << " variables and " << num_constraints << " constraints."
             << std::endl;
    }

    auto params = GlopParameters();
    // Try to make GLOP change the problem as little as possible,
    // so that we hopefully get a more precise solution
    params.set_primal_feasibility_tolerance(1e-14);
    params.set_drop_tolerance(0.0);
    params.set_solve_dual_problem(GlopParameters_SolverBehavior::GlopParameters_SolverBehavior_NEVER_DO);
    params.set_use_dual_simplex(false);

    LPSolver glop_solver;
    glop_solver.SetParameters(params);
    const auto status = glop_solver.Solve(lp);

    if (verbose) {
        std::cout << lp.GetPrettyProblemStats() << std::endl;
        std::cout << "Status: " << status << std::endl;
    }

    if (status == ProblemStatus::OPTIMAL) {
        for (int i = 0; i < num_variables; ++i) {
            x[i] = glop_solver.variable_values().at(variables.at(i));
        }

        return 0;
    } else {
        // TODO: Handle error conditions
        return 1;
    }
}