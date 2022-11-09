#include <iostream>
#include <time.h>
// #include <math.h>
#include <cmath>
#include <mpfr.h>
// #include <mpc.h>
#include <complex>
#include <vector>
#include <eigen3/unsupported/Eigen/MPRealSupport>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <mpreal.h>
#include <gmpxx.h>

/**
 * @brief Template class for precision numbers. The abstract class T should be a 
 * number type, generally either a primitive numeric type, mpf_class, or mpfr_real.
 * 
 * @tparam T Number type to use (double, mpf_class, mpfr::mpreal)
 */
template <class T>
class prec {
    protected:
        // Precision types
        using NReal = T;
        using NComplex = std::complex<T>;
        using NVector = std::vector<NComplex>;
        using NMatrix = Eigen::Matrix<NComplex, Eigen::Dynamic, Eigen::Dynamic>;
        using NArray = std::vector<NMatrix>;

        // Constants
        const NComplex ZERO;
        const NComplex ONE;
        const NComplex I;
        const NComplex TWO;
        const NComplex SQRT2;
        const NComplex PI;
    public:
        prec() : 
            ZERO                    (NComplex{0., 0.}), 
            ONE                     (NComplex{1., 0.}), 
            I                       (NComplex{0., 1.}), 
            TWO                     (NComplex{2., 0.}), 
            SQRT2                   (sqrt(TWO)), 
            PI                      (NComplex{M_PI, 0.}) 
        {};
        NComplex get_zero()         {return ZERO;}
        NComplex get_one()          {return ONE;}
        NComplex get_i()            {return I;}
        NComplex get_two()          {return TWO;}
        NComplex get_sqrt2()        {return SQRT2;}
        NComplex get_pi()           {return PI;};
};

// Specialized constructors for precision type when T is mpf_class. 
template<> 
prec<mpf_class>::prec() : 
    ZERO            (NComplex{mpf_class("0"), mpf_class("0")}), 
    ONE             (NComplex{mpf_class("1"), mpf_class("0")}), 
    I               (NComplex{mpf_class("0"), mpf_class("1")}), 
    TWO             (NComplex{mpf_class("2"), mpf_class("0")}),
    SQRT2           (NComplex{sqrt(TWO.real()), mpf_class("0")}),
    PI              (NComplex{mpf_class("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006"), mpf_class("0")})    // http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
{};

template<> 
prec<mpfr::mpreal>::prec() : 
    ZERO            (NComplex{mpfr::mpreal("0.0"), mpfr::mpreal("0.0")}), 
    ONE             (NComplex{mpfr::mpreal("1.0"), mpfr::mpreal("0.0")}), 
    I               (NComplex{mpfr::mpreal("0.0"), mpfr::mpreal("1.0")}), 
    TWO             (NComplex{mpfr::mpreal("2.0"), mpfr::mpreal("0.0")}),
    SQRT2           (NComplex{sqrt(TWO.real()), mpfr::mpreal("0.0")}),
    PI              (NComplex{mpfr::const_pi(), mpfr::mpreal("0.0")})
{};

const mpfr_prec_t PRECISION    = mpfr_prec_t(128);
const mpfr_rnd_t RRND          = MPFR_RNDN;

/**
 * @brief Container for imaginary domain (Euclidean frequency) data.
 * 
 * @tparam T Precision type to use. 
 */
template <class T>
class ImaginaryDomainData : prec<T> {

    // We can declare these first since they always have to be in each definition, then use 
    // another private section later to declare the fields. 
    private:
        using typename prec<T>::NReal;
        using typename prec<T>::NComplex;
        using typename prec<T>::NVector;

        NVector freqs;
        NVector ng;
        NVector h;

    public:
        ImaginaryDomainData(
            const NVector freq0,
            const NVector ng0
        );
        NVector get_freqs()          {return freqs;}
        NVector get_ng()            {return ng;}
        NVector get_h()             {return h;}

        // Utility functions for Mobius transformation.
        NVector mobius(const NVector z);
        NVector inv_mobius(const NVector z);
        NVector get_pick_matrix(NVector freqs, NVector ng);

};

/**
 * @brief Container for real domain data. 
 * 
 * @tparam T Precision type to use. 
 */
template <class T>
class RealDomainData : prec<T> {

    private:
        using typename prec<T>::NReal;
        using typename prec<T>::NComplex;
        using typename prec<T>::NVector;
    
    public:
        RealDomainData(
            int start = 0,
            int stop = 1,
            int num = 50, 
            double eta = 0.001
        );
    
    private:
        NVector freq;
        NVector rho;

};

/**
 * Container class for the Schur interpolation algorithm.
 * 
 */
template <class T>
class Schur : prec<T> {

    private:
        using typename prec<T>::NReal;
        using typename prec<T>::NComplex;
        using typename prec<T>::NVector;
    
    public:
        Schur(
            ImaginaryDomainData<T> imag
        );
        NVector generate_phis();

    private:
        ImaginaryDomainData<T> imag;
        NVector phi;
        int npts;

};

template <class T>
class Nevanlinna : prec<T> {
    
    private:
        using typename prec<T>::NReal;
        using typename prec<T>::NComplex;
        using typename prec<T>::NVector;
        using typename prec<T>::NMatrix;
        using typename prec<T>::NArray;

    public:
        Nevanlinna(
            NVector matsubara,
            NVector ng
        );
        void evaluate(
            int start = 0,
            int stop = 1,
            int num = 50, 
            double eta = 0.001,
            bool map_back = true
        );

    private: 
        Schur<T> schur;

};