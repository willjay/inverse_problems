#include <iostream>
#include <omp.h>
#include <chrono>

#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

#include "tests.hpp"

void eigen_tests() {

    Prec<double>::NMatrix my_mat = Prec<double>::NMatrix::Zero(2, 2);
    my_mat(0, 1) = Prec<double>::get_two() * Prec<double>::get_pi();
    std::cout << my_mat << std::endl;
    print_matrix<double>(my_mat);
    std::cout << my_mat.row(0) << std::endl;

    // Prec<double>::NMatrix a = Prec<double>::NMatrix::Zero(2, 2);
    Prec<double>::NMatrix a (2, 2);
    a << Prec<double>::NComplex{1.0, 0.0}, Prec<double>::NComplex{0.0, 1.0},
         Prec<double>::NComplex{0.0, 0.0}, Prec<double>::NComplex{0.0, 2.0};
    Prec<double>::NMatrix b (2, 2);
    b << Prec<double>::NComplex{2.0, 0.0}, Prec<double>::NComplex{0.0, 0.0},
         Prec<double>::NComplex{0.0, 1.0}, Prec<double>::NComplex{1.0, 0.0};
    std::cout << "a" << std::endl;
    print_matrix<double>(a);
    std::cout << "b" << std::endl;
    print_matrix<double>(b);
    std::cout << "a * b" << std::endl;
    print_matrix<double>(a*b);
    std::cout << "b * a" << std::endl;
    print_matrix<double>(b*a);

}

int main(int argc, char const *argv[]) {

    // Initialize precision
    mpf_set_default_prec(PRECISION);
    std::cout.precision(DIGITS);
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(DIGITS));

    std::cout << Prec<double>::get_i() << std::endl;
    std::cout << Prec<mpfr::mpreal>::get_pi() << std::endl;
    std::cout << Prec<mpfr::mpreal>::get_sqrt2() << std::endl;

    Prec<double>::NComplex z = {0.0, 0.2};
    std::cout << z << std::endl;

    // Sample Matsubara frequencies
    Prec<double>::NVector ff = {
        {0.0, 1.0},
        {0.0, 2.0},
        {0.0, 3.0},
        {0.0, 4.0},
        {0.0, 5.0}
    };
    std::cout << ff[0].real() << std::endl;
    std::cout << ff[0] << std::endl;

    // Sample Green's functions
    Prec<double>::NVector ng_pts = {
        {0.2, 0.4},
        {0.1, 0.92},
        {-0.43, 0.03},
        {-0.8, 0.23},
        {0.2, 0.1}
    };

    std::cout << "Printing vec" << std::endl;
    print_vector<double>(ng_pts);
    ImaginaryDomainData<double> imag (ff, ng_pts);
    print_vector<double>(imag.get_lambda());
    Prec<double>::NVector inv_h = Nevanlinna<double>::inv_mobius(imag.get_lambda());
    print_vector<double>(inv_h);

    // the "NVector" typedef is the same in each of the daughter classes, regardless of where you call it
    // Generally just use Prec<T> whenever possible to avoid confusion.
    Prec<double>::NVector tmp ({{0.0, 0.2}});
    // ImaginaryDomainData<double>::NVector tmp ({{0.0, 0.2}});
    Prec<double>::NVector htmp = Nevanlinna<double>::mobius(tmp);
    print_vector<double>(htmp);

    Prec<double>::NMatrix pick_mat = Schur<double>::get_pick_realspace(ff, ng_pts);
    print_matrix<double>(pick_mat);

    eigen_tests();

    Schur<double> schur (imag);
    std::cout << "phi:" << std::endl;
    print_vector<double>(schur.get_phi());

    Prec<double>::NVector z_eval_dbl ({
        {1, 0.001},
        {2, 0.001},
        {3, 0.001}
    });
    std::cout << "printing continuation with double prec" << std::endl;
    print_vector<double>(schur.eval_interp(z_eval_dbl));

    // do it again with mpreal
    Prec<mpfr::mpreal>::NVector ff_mpfr = {
        {mpfr::mpreal("0.0"), mpfr::mpreal("1.0")},
        {mpfr::mpreal("0.0"), mpfr::mpreal("2.0")},
        {mpfr::mpreal("0.0"), mpfr::mpreal("3.0")},
        {mpfr::mpreal("0.0"), mpfr::mpreal("4.0")},
        {mpfr::mpreal("0.0"), mpfr::mpreal("5.0")}
    };
    Prec<mpfr::mpreal>::NVector ng_pts_mpfr = {
        {mpfr::mpreal("0.2"), mpfr::mpreal("0.4")},
        {mpfr::mpreal("0.1"), mpfr::mpreal("0.92")},
        {mpfr::mpreal("-0.43"), mpfr::mpreal("0.03")},
        {mpfr::mpreal("-0.8"), mpfr::mpreal("0.23")},
        {mpfr::mpreal("0.2"), mpfr::mpreal("0.1")}
    };
    ImaginaryDomainData<mpfr::mpreal> imag_mpfr (ff_mpfr, ng_pts_mpfr);
    Schur<mpfr::mpreal> schur_mpfr (imag_mpfr);
    Prec<mpfr::mpreal>::NVector z_eval_mpfr ({
        {mpfr::mpreal("1"), mpfr::mpreal("0.001")},
        {mpfr::mpreal("2.0123456789123456789123456789123456789123456789123456789123456789123456789"), 
         mpfr::mpreal("0.00123456789123456789123456789123456789123456789123456789123456789123456789")},
        {mpfr::mpreal("3"), mpfr::mpreal("0.001")}
    });
    std::cout << "printing continuation with mpreal prec" << std::endl;
    print_vector<mpfr::mpreal>(schur_mpfr.eval_interp(z_eval_mpfr));

    // Test real and imag

    // std::cout.precision(50);

    std::cout << "testing vec_to_str" << std::endl;
    std::vector<std::string> rstr = vec_to_rstring<mpfr::mpreal>(z_eval_mpfr);
    std::vector<std::string> istr = vec_to_istring<mpfr::mpreal>(z_eval_mpfr);
    std::cout << rstr[1] << std::endl;
    std::cout << istr[1] << std::endl;

    // Testing precision of Pi
    std::cout << mpfr::const_pi(mpfr::digits2bits(100)) << std::endl;
    std::cout << mpfr::const_pi(mpfr::digits2bits(DIGITS)) << std::endl;
    std::cout << mpfr::const_pi() << std::endl;

    // TODO figure out how to make constant PI accurate
    std::cout << Prec<mpfr::mpreal>::PI << std::endl;
    std::cout << Prec<mpfr::mpreal>::PI.real().toString(DIGITS) << std::endl;

    // Test precision of constants to 32 digits
    std::cout << Prec<mpfr::mpreal>::ONE << std::endl;
    std::cout << Prec<mpfr::mpreal>::I << std::endl;
    std::cout << Prec<mpfr::mpreal>::TWO << std::endl;
    std::cout << Prec<mpfr::mpreal>::SQRT2 << std::endl;
    std::cout << Prec<mpfr::mpreal>::DEFAULT_ETA << std::endl;
    std::cout << Prec<mpfr::mpreal>::EPSILON << std::endl;

    // Test array overload
    RealDomainData<double> om;
    Prec<double>::NComplex this_elem = om[3];
    std::cout << om[3] << std::endl;
    om[3] = Prec<double>::NComplex{10.0, -9.2};
    std::cout << om[3] << std::endl;

    return 0;
}