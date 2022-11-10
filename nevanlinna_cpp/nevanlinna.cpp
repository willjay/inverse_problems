#include <iostream>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Cholesky>
#include "nevanlinna.hpp"
#include <omp.h>
#include <chrono>

template <class T>
void print_complex(const typename prec<T>::NComplex c) {
    std::cout << c.real() << " + " << c.imag() << "i";
}

template <class T>
void print_vector(const typename prec<T>::NVector v) {
    std::cout << "[";
    for (int ii = 0; ii < v.size(); ii++) {
        print_complex<T>(v[ii]);
        if (ii < v.size() - 1) {
            std::cout << ", ";
        } else {
            std::cout << "]" << std::endl;
        }   
    }
}

template <class T>
ImaginaryDomainData<T>::ImaginaryDomainData(const NVector freqs0, const NVector ng0) : 
    freqs(freqs0), ng(ng0), h(mobius(ng)) 
{};

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

    prec<double> x1;
    prec<mpf_class> x2;
    prec<mpfr::mpreal> x3;
    std::cout << x2.get_pi() << std::endl;
    std::cout << x3.get_pi() << std::endl;

    // ImaginaryDomainData<double> imag ();
    prec<double>::NComplex z = {0.0, 0.2};
    std::cout << z << std::endl;

    prec<double>::NVector ff = {
        {0.0, 1.0},
        {0.0, 2.0},
        {0.0, 3.0}
    };
    std::cout << ff[0].real() << std::endl;
    std::cout << ff[0] << std::endl;

    prec<double>::NVector ng_pts = {
        {0.2, 0.8},
        {0.1, 0.92},
        {-0.43, 0.03}
    };

    std::cout << "Printing vec" << std::endl;
    print_vector<double>(ng_pts);
    ImaginaryDomainData<double> imag (ff, ng_pts);
    print_vector<double>(imag.get_h());

}

int main(int argc, char const *argv[]) {
    mpf_set_default_prec(PRECISION);
    prec<mpf_class> precision;

    playground();

    return 0;
}