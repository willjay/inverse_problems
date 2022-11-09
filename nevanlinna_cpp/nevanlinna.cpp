#include <iostream>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Cholesky>
#include "nevanlinna.hpp"
#include <omp.h>
#include <chrono>

// This breaks because Eigen doesn't seem to work well with mpc
// VectorComplex h(const VectorComplex& z) {
//     VectorComplex out;
//     out = (z + I) / (z - I);
//     return out;
// }

template <class T>
ImaginaryDomainData<T>::ImaginaryDomainData(const NVector freqs0, const NVector ng0) : freqs(freqs0), ng(ng0) {
    
};

// template <class T>
// NVector ImaginaryDomainData<T>::mobius(const NVector z) {

// ImaginaryDomainData<T>::NVector ImaginaryDomainData<T>::mobius(const NVector z) {
template <class T>
typename ImaginaryDomainData<T>::NVector ImaginaryDomainData<T>::mobius(const NVector z) {
    NVector hz(z);
    for (int ii = 0; ii < z.size(); ii++) {
        hz[ii] = (z[ii] - this->I) / (z[ii] + this->I);
    }
    return hz;
}

// template <class T>
// ImaginaryDomainData<T>::NVector ImaginaryDomainData<T>::inv_mobius(const NVector z) {
//     NVector hinvz(z);
//     for (int ii = 0; ii < z.size(); ii++) {
//         hinvz[ii] = prec.get_i() * (prec.get_one() + z[ii]) / (prec.get_one() - z[ii]);
//     }
//     return hinvz;
// }

// NVector get_pick_matrix(NVector freq, NVector ng);

// Random testing ground.
void playground() {

    std::cout << "hello world" << std::endl;
    // mpf_class I = 0.;
    // std::cout << "I is: " << I << std::endl;

    prec<double> x;
    prec<mpf_class> y;
    prec<mpfr::mpreal> z;
    std::cout << y.get_pi() << std::endl;
    std::cout << z.get_pi() << std::endl;

    // ImaginaryDomainData<double> imag ();

}

int main(int argc, char const *argv[]) {
    mpf_set_default_prec(PRECISION);
    prec<mpf_class> precision;

    playground();

    return 0;
}