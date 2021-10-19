use fenris::solid::materials::{YoungPoisson, LameParameters};
use matrixcompare::assert_scalar_eq;

fn check_effective_parameters(effective_params: YoungPoisson<f64>, paper_params: YoungPoisson<f64>) {
    let LameParameters { mu: mu_eff, .. } = effective_params.into();
    let LameParameters { lambda: lambda_bad, mu } = paper_params.into();

    let lambda_eff = 2.0 * (mu_eff * effective_params.poisson) / (1.0 - 2.0 * effective_params.poisson);
    assert_scalar_eq!(mu_eff, mu, comp = abs, tol = mu_eff * 1e-12);
    assert_scalar_eq!(lambda_eff, lambda_bad, comp=abs, tol = lambda_eff * 1e-9);
}

#[test]
fn errata_conversion_formulas() {
    // Test that the formulas for the "effective" material parameters given in the errata are correct.
    // We do this by checking that, using these formulas, we obtain the same Lame parameters as our
    // incorrect implementation

    // Numerical verification (Section 5.4)
    {
        let effective_params = YoungPoisson {
            young: 2.67857142857e6,
            poisson: 0.25
        };
        let paper_params = YoungPoisson {
            young: 3e6,
            poisson: 0.4
        };
        check_effective_parameters(effective_params, paper_params);
    }

    // Twisting cylinder
    {
        let effective_params = YoungPoisson {
            young: 4.82625482625e6,
            poisson: 0.42857142857
        };
        let paper_params = YoungPoisson {
            young: 5e6,
            poisson: 0.48
        };
        check_effective_parameters(effective_params, paper_params);
    }

    // Armadillo slingshot
    {
        let effective_params = YoungPoisson {
            young: 4.46428571429e5,
            poisson: 0.25
        };
        let paper_params = YoungPoisson {
            young: 5e5,
            poisson: 0.4
        };
        check_effective_parameters(effective_params, paper_params);
    }
}