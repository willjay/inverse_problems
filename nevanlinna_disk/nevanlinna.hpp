#ifndef NEVANLINNA_H
#define NEVANLINNA_H

#include <iostream>
#include <time.h>
#include <cmath>
#include <tuple>
// #include <mpfr.h>
#include <complex>
#include <vector>
#include <eigen3/unsupported/Eigen/MPRealSupport>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Cholesky>
#include <mpreal.h>
#include <gmpxx.h>
#include <H5Cpp.h>

#include <boost/math/differentiation/autodiff.hpp>

#include "prec.hpp"

// ************************************************************ //
// ********************* Template Classes ********************* //
// ************************************************************ //
// Note that templated classes must be implemented in the same .hpp file

/**
 * @brief Container for imaginary domain (Euclidean frequency) data.
 * 
 * @tparam T Precision type to use. 
 */
template <class T>
class ImaginaryDomainData : Prec<T> {

    private:
        using typename Prec<T>::NReal;
        using typename Prec<T>::NComplex;
        using typename Prec<T>::NVector;

        NVector freqs;          // Matsubara frequencies
        NVector xi;             // Mobius transform of Matsubara frequencies.
        NVector ng;             // Negative of Green's function (nevanlinna)
        NVector lam;            // Mobius transform of NG.
        int npts;

    public:
        ImaginaryDomainData(
            const NVector& freq0,
            const NVector& ng0
        );

        // Accessors
        NVector get_freqs() const;
        NVector get_xi() const;
        NVector get_ng() const;
        NVector get_lambda() const;
        int get_npts() const;
        void set_freqs(const NVector& new_freqs);
        void set_ng(const NVector& new_ng);

};

/**
 * @brief Container for real domain data. 
 * 
 * @tparam T Precision type to use. 
 */
template <class T>
class RealDomainData : Prec<T> {

    private:
        using typename Prec<T>::NReal;
        using typename Prec<T>::NComplex;
        using typename Prec<T>::NVector;

        NVector freqs;
        NVector rho;
        int npts;
    
    public:
        // Default arguments go in the header file
        RealDomainData(
            double start = 0.0,
            double stop = 1.0,
            int num = 100,
            NReal eta = Prec<T>::DEFAULT_ETA
        );
        NComplex operator[] (int i) const;
        NComplex& operator[] (int i);

        // Accessors
        NVector get_freqs() const;
        NVector get_rho() const;
        int get_npts() const;
        void set_freqs(const NVector& new_freqs);
        void set_rho(const NVector& new_rho);

};

/**
 * Container class for the Schur interpolation algorithm. Note this algorithm reconstructs a function 
 * D --> D, where D is the unit disk. 
 * 
 */
template <class T>
class Schur : Prec<T> {

    private:
        using typename Prec<T>::NReal;
        using typename Prec<T>::NComplex;
        using typename Prec<T>::NVector;
        using typename Prec<T>::NMatrix;

        int npts;
        ImaginaryDomainData<T> imag;
        NVector w;                                    // w_k^{(k-1)} of the paper

        NMatrix U_matrix(NComplex z, int k);            // U_k matrix of the paper
    
    public:
        Schur(
            const ImaginaryDomainData<T>& imag0
        );

        // Accessors
        ImaginaryDomainData<T> get_imag() const;
        NVector get_w() const;
        int get_npts() const;

        void set_imag(const ImaginaryDomainData<T>& new_imag);
        void set_w(const NVector& new_w);

        // Static methods
        static NMatrix get_pick(const NVector& yvals, const NVector& lambda_vals);
        static NMatrix get_pick_realspace(const NVector& freqs, const NVector& ng);
        static NComplex zero_fcn(const NComplex& z);

        // Utility methods
        NVector generate_ws();
        std::tuple<NVector, NVector, NVector, NVector, NVector> eval_interp(const NVector& z, NComplex (*fn)(const NComplex&) = &zero_fcn);

};

template <class T>
class Nevanlinna : Prec<T> {
    
    private:
        using typename Prec<T>::NReal;
        using typename Prec<T>::NComplex;
        using typename Prec<T>::NVector;
        using typename Prec<T>::NMatrix;
        using typename Prec<T>::NArray;

        NVector P;      // Nevanlinna coeffs
        NVector Q;
        NVector R;
        NVector S;

        Schur<T> schur;

    public:
        Nevanlinna(
            NVector& matsubara,
            NVector& ng
        );
        std::tuple<RealDomainData<T>, NVector> evaluate(
            double start = 0.0,
            double stop = 1.0,
            int num = 100, 
            NReal eta = Prec<T>::DEFAULT_ETA
        );
        
        // Accessors
        Schur<T> get_schur() const;
        NVector get_P() const;
        NVector get_Q() const;
        NVector get_R() const;
        NVector get_S() const;

        void set_schur(const Schur<T>& new_schur);
        void set_P(const NVector& new_P);
        void set_Q(const NVector& new_Q);
        void set_R(const NVector& new_R);
        void set_S(const NVector& new_S);
        
        // Static methods
        static NVector mobius(const NVector& z);
        static NVector inv_mobius(const NVector& z);

};

template <class T>
class H5Reader : Prec<T> {
    
    private:

        using typename Prec<T>::NComplex;
        using typename Prec<T>::NVector;

        std::string h5_path;
        std::vector<std::string> freq_str;
        std::vector<std::string> ngr_str;
        std::vector<std::string> ngi_str;

        int npts;
        int beta;
        NVector freqs;
        NVector ng;

        int read_int(std::string dset_path);
        std::vector<std::string> read_field(std::string dset_path);

    public:
        H5Reader(std::string fname);

        // Accessors
        int get_npts() const;       // Different than beta if we subsample
        int get_beta() const;
        std::vector<std::string> get_freqs_str() const;
        std::vector<std::string> get_ng_real_str() const;
        std::vector<std::string> get_ng_imag_str() const;
        NVector get_freqs() const;
        NVector get_ng() const;

};

template <class T>
class H5Writer : Prec<T> {

    private:

        using typename Prec<T>::NReal;
        using typename Prec<T>::NComplex;
        using typename Prec<T>::NVector;
    
        std::string h5_path;
        H5::H5File* f;
        int beta;
        double start;
        double stop;
        int num;
        NReal eta;

        NVector freqs;
        NVector xi_list;

        NVector ng;
        NVector lambda;

        NVector w_list;
        NVector recon;

        Nevanlinna<T> nev;

        void write_int(std::string dset_path, int data);
        void write_double(std::string dset_path, double data);
        void write_nreal(std::string dset_path, NReal data);
        void write_nvector(std::string dset_path, NVector data);

    public:
        H5Writer(
            std::string fname, 
            int beta0,
            double start0, 
            double stop0, 
            int num0, 
            const NReal& eta0,
            const NVector& freqs0, 
            const NVector& ng0,
            const NVector& recon0,
            const Nevanlinna<T>& nev0
        );

        // Accessors
        std::string get_fname() const;
        int get_beta() const;
        double get_start() const;
        double get_stop() const;
        int get_num() const;
        NReal get_eta() const;
        NVector get_freqs() const;
        NVector get_ng() const;
        NVector get_w_list() const;
        NVector get_recon() const;
        // Schur get_schur() const;

        void write();

};

// ************************************************************ //
// ************************* Accessors ************************ //
// ************************************************************ //

template <class T>
typename ImaginaryDomainData<T>::NVector ImaginaryDomainData<T>::get_freqs() const {
    return freqs; 
}

template <class T>
typename ImaginaryDomainData<T>::NVector ImaginaryDomainData<T>::get_xi() const {
    return xi; 
}

template <class T>
typename ImaginaryDomainData<T>::NVector ImaginaryDomainData<T>::get_ng() const {
    return ng;
}

template <class T>
typename ImaginaryDomainData<T>::NVector ImaginaryDomainData<T>::get_lambda() const {
    return lam;
}

template <class T>
int ImaginaryDomainData<T>::get_npts() const {
    return npts;
}

template <class T>
void ImaginaryDomainData<T>::set_freqs(const NVector& new_freqs) { 
    freqs = new_freqs;
    xi = Nevanlinna<T>::mobius(freqs);
    npts = freqs.size();
}

template <class T>
void ImaginaryDomainData<T>::set_ng(const NVector& new_ng) { 
    ng = new_ng;
    lam = Nevanlinna<T>::mobius(new_ng);
}

template <class T>
typename RealDomainData<T>::NVector RealDomainData<T>::get_freqs() const {
    return freqs;
}

template <class T>
typename RealDomainData<T>::NVector RealDomainData<T>::get_rho() const {
    return rho;
}

template <class T>
int RealDomainData<T>::get_npts() const {
    return npts;
}

template <class T>
void RealDomainData<T>::set_freqs(const NVector& new_freqs) { 
    freqs = new_freqs;
    npts = freqs.size();
}

template <class T>
void RealDomainData<T>::set_rho(const NVector& new_rho) { 
    rho = new_rho;
}

template <class T>
ImaginaryDomainData<T> Schur<T>::get_imag() const {
    return imag;
}

template <class T>
typename Schur<T>::NVector Schur<T>::get_w() const {
    return w;
}

template <class T>
int Schur<T>::get_npts() const {
    return npts;
}

template <class T>
void Schur<T>::set_imag(const ImaginaryDomainData<T>& new_imag) {
    imag = new_imag;
}

template <class T>
void Schur<T>::set_w(const NVector& new_w) {
    w = new_w; 
    npts = new_w.size();
}

template <class T>
Schur<T> Nevanlinna<T>::get_schur() const {
    return schur;
}

template <class T>
typename Nevanlinna<T>::NVector Nevanlinna<T>::get_P() const {
    return P;
}

template <class T>
typename Nevanlinna<T>::NVector Nevanlinna<T>::get_Q() const {
    return Q;
}

template <class T>
typename Nevanlinna<T>::NVector Nevanlinna<T>::get_R() const {
    return R;
}

template <class T>
typename Nevanlinna<T>::NVector Nevanlinna<T>::get_S() const {
    return S;
}

template <class T>
void Nevanlinna<T>::set_schur(const Schur<T>& new_schur) {
    schur = new_schur;
}

template <class T>
void Nevanlinna<T>::set_P(const NVector& new_P) {
    P = new_P;
}

template <class T>
void Nevanlinna<T>::set_Q(const NVector& new_Q) {
    Q = new_Q;
}

template <class T>
void Nevanlinna<T>::set_R(const NVector& new_R) {
    R = new_R;
}

template <class T>
void Nevanlinna<T>::set_S(const NVector& new_S) {
    S = new_S;
}

template <class T>
int H5Reader<T>::get_npts() const {
    return npts;
}

template <class T>
int H5Reader<T>::get_beta() const {
    return beta;
}

template <class T>
std::vector<std::string> H5Reader<T>::get_freqs_str() const {
    return freq_str; 
}

template <class T>
std::vector<std::string> H5Reader<T>::get_ng_real_str() const {
    return ngr_str; 
}

template <class T>
std::vector<std::string> H5Reader<T>::get_ng_imag_str() const {
    return ngi_str;
}

template <class T>
typename H5Reader<T>::NVector H5Reader<T>::get_freqs() const {
    return freqs;
}

template <class T>
typename H5Reader<T>::NVector H5Reader<T>::get_ng() const {
    return ng;
}

template <class T>
std::string H5Writer<T>::get_fname() const {
    return h5_path;
}

template <class T>
int H5Writer<T>::get_beta() const {
    return beta;
}

template <class T>
double H5Writer<T>::get_start() const {
    return start;
}

template <class T>
double H5Writer<T>::get_stop() const {
    return stop;
}

template <class T>
int H5Writer<T>::get_num() const {
    return num;
}

template <class T>
typename H5Writer<T>::NReal H5Writer<T>::get_eta() const {
    return eta;
}

template <class T>
typename H5Writer<T>::NVector H5Writer<T>::get_freqs() const {
    return freqs;
}

template <class T>
typename H5Writer<T>::NVector H5Writer<T>::get_ng() const {
    return ng;
}

template <class T>
typename H5Writer<T>::NVector H5Writer<T>::get_w_list() const {
    return w_list;
}

template <class T>
typename H5Writer<T>::NVector H5Writer<T>::get_recon() const {
    return recon;
}

// ************************************************************ //
// ************************ Constructors ********************** //
// ************************************************************ //

template <class T>
ImaginaryDomainData<T>::ImaginaryDomainData(const NVector& freqs0, const NVector& ng0) : 
    freqs(freqs0), xi(Nevanlinna<T>::mobius(freqs0)), ng(ng0), lam(Nevanlinna<T>::mobius(ng)), npts(freqs0.size()) {}

template <class T>
RealDomainData<T>::RealDomainData(double start, double stop, int num, NReal eta) : freqs(num), npts(num) {
    T start_T = static_cast<T>(start);
    T delta = static_cast<T>(stop - start) / static_cast<T>(num - 1);
    for (int i = 0; i < num; i++) {
        freqs[i] = NComplex{start_T + i * delta, eta};
    }
}

template <class T>
Schur<T>::Schur(const ImaginaryDomainData<T>& imag0) : npts(imag0.get_npts()), imag(imag0), w(npts) {
    generate_ws();
}

template <class T>
Nevanlinna<T>::Nevanlinna(NVector& matsubara, NVector& ng) : schur(ImaginaryDomainData<T>(matsubara, ng)), \
        P(1), Q(1), R(1), S(1) {}

template <class T>
H5Reader<T>::H5Reader(std::string fname) : 
    h5_path (fname), freq_str (read_field("freqs/imag")), ngr_str (read_field("ng/real")), ngi_str (read_field("ng/imag")), 
    npts (freq_str.size()), beta (read_int("beta")), freqs (npts), ng (npts) 
{
    for (int ii = 0; ii < npts; ii++) {
        freqs[ii] = NComplex{"0", freq_str[ii]};
        ng[ii]    = NComplex{ngr_str[ii], ngi_str[ii]};
    }
}

template <class T>
H5Writer<T>::H5Writer(std::string fname, int beta0, double start0, double stop0, int num0, const NReal& eta0, 
                        const NVector& freqs0, const NVector& ng0, const NVector& recon0, const Nevanlinna<T>& nev0)
    : h5_path (fname),  beta(beta0), start(start0), stop(stop0), num(num0), eta(eta0), freqs(freqs0), xi_list(Nevanlinna<T>::mobius(freqs0)), 
      ng(ng0), lambda(Nevanlinna<T>::mobius(ng0)), recon(recon0), w_list(nev0.get_schur().get_w()), nev(nev0) {
    f = new H5::H5File( h5_path, H5F_ACC_TRUNC );
}

// ************************************************************ //
// ************************** Methods ************************* //
// ************************************************************ //

template <class T>
typename RealDomainData<T>::NComplex RealDomainData<T>::operator[](int i) const {
    return freqs[i];
}

template <class T>
typename RealDomainData<T>::NComplex& RealDomainData<T>::operator[](int i) {
    return freqs[i];
}

// Forms the U matrix U_k(xi)
template <class T>
typename Schur<T>::NMatrix Schur<T>::U_matrix(NComplex z, int k) {
    NVector xi = imag.get_xi();       // xi = h(y), i.e. Mobius transform of Matsubara freqs. Note these should all be purely real (TODO check)

    NComplex blashke = (xi[k] - z) / (Prec<T>::ONE - std::conj(xi[k]) * z);
    NMatrix U (2, 2);
    U << blashke, w[k],
         std::conj(w[k]) * blashke, Prec<T>::ONE;
    U = U / std::sqrt(Prec<T>::ONE - w[k] * std::conj(w[k]));
    return U;
}

/**
 * @brief Generate w[j] parameters inductively.
 * 
 * @tparam T Precision type to use.
 * @return Schur<T>::NVector Result for w.
 */
template <class T>
typename Schur<T>::NVector Schur<T>::generate_ws() {
    NVector xi = imag.get_xi();
    NVector lam = imag.get_lambda();
    w[0] = lam[0];
    for (int j = 1; j < npts; j++) {
        NMatrix arr = NMatrix::Identity(2, 2);
        for (int k = 0; k < j; k++) {
            arr = arr * U_matrix(xi[j], k);
        }

        NComplex num = lam[j] * arr(1, 1) - arr(0, 1);
        NComplex denom = arr(0, 0) - lam[j] * arr(1, 0);
        if (is_zero<T>(num)) {
            w[j] = Prec<T>::ZERO;
        } else {
            w[j] = num / denom;
        }
    }
    return w;
}

/**
 * @brief Evaluates the Schur interpolant and analytically continues from the measured points to the 
 * specified points in the unit disk.
 * 
 * @tparam T Precision type to use.
 * @param z Points to evaluate the interpolant at.
 * @param fn theta_{k + 1} function.
 * @return Schur<T>::NVector 
 */
template <class T>
std::tuple<typename Schur<T>::NVector, typename Schur<T>::NVector, typename Schur<T>::NVector, typename Schur<T>::NVector, typename Schur<T>::NVector> Schur<T>::eval_interp(const NVector& z, NComplex (*fn)(const NComplex&)) {
    int n_eval = z.size();
    NVector interp (n_eval);
    NMatrix ncoeffs (2, 2);
    NMatrix factor (2, 2);
    NVector p (n_eval);
    NVector q (n_eval);
    NVector r (n_eval);
    NVector s (n_eval);
    for (int i = 0; i < z.size(); i++) {
        NComplex zval = z[i];
        ncoeffs = NMatrix::Identity(2, 2);

        for (int k = 0; k < npts; k++) {
            ncoeffs = ncoeffs * U_matrix(zval, k);
        }
        p[i] = ncoeffs(0, 0);
        q[i] = ncoeffs(0, 1);
        r[i] = ncoeffs(1, 0);
        s[i] = ncoeffs(1, 1);

        NComplex num = ncoeffs(0, 0) * fn(zval) + ncoeffs(0, 1);
        NComplex denom = ncoeffs(1, 0) * fn(zval) + ncoeffs(1, 1);
        if (is_zero<T>(num)) {
            // interp[i] = NComplex{"0", "0"};
            interp[i] = Prec<T>::ZERO;
        } else {
            interp[i] = num / denom;
        }
        interp[i] = num / denom;
    }
    // if (map_back) {
    //     return std::make_tuple(Nevanlinna<T>::inv_mobius(interp), a, b, c, d);
    // }
    return std::make_tuple(interp, p, q, r, s);
}

/**
 * @brief Evaluates the Nevanlinna interpolant to solve the interpolation problem on the disk D --> D. 
 * 
 * @tparam T 
 * @param start 
 * @param stop 
 * @param num 
 * @param eta 
 * @return std::tuple<RealDomainData<T>, typename Nevanlinna<T>::NVector> 
 */
template <class T>
std::tuple<RealDomainData<T>, typename Nevanlinna<T>::NVector> Nevanlinna<T>::evaluate(double start, double stop, int num, NReal eta) {

    // TODO make input function a parameter here

    RealDomainData<T> omega (start, stop, num, eta);
    // NVector freqs = omega.get_freqs();
    NVector freqs = Nevanlinna<T>::mobius(omega.get_freqs());
    NVector interp;
    std::tie(interp, P, Q, R, S) = schur.eval_interp(freqs, Schur<T>::zero_fcn);

    return std::make_tuple(omega, interp);
}

template <class T>
int H5Reader<T>::read_int(std::string dset_path) {
    H5::H5File f (h5_path, H5F_ACC_RDONLY);
    H5::DataSet dset = f.openDataSet(dset_path);
    H5::DataSpace dspace = dset.getSpace();
    int data_out[1];
    data_out[0] = 0;
    dset.read(data_out, H5::PredType::NATIVE_INT, dspace);
    int output = data_out[0];

    return output;
}

template <class T>
std::vector<std::string> H5Reader<T>::read_field(std::string dset_path) {
    H5::H5File f (h5_path, H5F_ACC_RDONLY);
    H5::DataSet dset = f.openDataSet(dset_path);
    H5::DataSpace dspace = dset.getSpace();
    hsize_t rank;
    hsize_t dims[2];  
    rank = dspace.getSimpleExtentDims(dims, nullptr);
    std::vector<char*> data;
    data.resize(dims[0]);
    H5::StrType string_type = dset.getStrType();
    dset.read(data.data(), string_type, dspace);
    std::vector<std::string> output (data.begin(), data.end());

    return output;
}

template <class T>
void H5Writer<T>::write() {
    this->write_int("beta", beta);
    this->write_int("num", num);

    this->write_double("start", start);
    this->write_double("stop", stop);

    this->write_nreal("eta", eta);

    this->write_nvector("freqs", freqs);
    this->write_nvector("xi", xi_list);
    this->write_nvector("ng", ng);
    this->write_nvector("lambda", lambda);

    this->write_nvector("w", w_list);
    this->write_nvector("recon", recon);

    this->write_nvector("P", nev.get_P());
    this->write_nvector("Q", nev.get_Q());
    this->write_nvector("R", nev.get_R());
    this->write_nvector("S", nev.get_S());

}

template <class T>
void H5Writer<T>::write_int(std::string dset_path, int data) {
    hsize_t dimsf[1] { 1 };
    H5::DataSpace dataspace (1, dimsf);
    H5::DataSet dset = f->createDataSet(dset_path, H5::PredType::NATIVE_INT, dataspace);
    dset.write(&data, H5::PredType::NATIVE_INT);
}

template <class T>
void H5Writer<T>::write_double(std::string dset_path, double data) {
    hsize_t dimsf[1] { 1 };
    H5::DataSpace dataspace (1, dimsf);
    H5::DataSet dset = f->createDataSet(dset_path, H5::PredType::NATIVE_DOUBLE, dataspace);
    dset.write(&data, H5::PredType::NATIVE_DOUBLE);
}

template <class T>
void H5Writer<T>::write_nreal(std::string dset_path, NReal data) {
    std::string tmp_str = data.toString(DIGITS);
    const char* s = tmp_str.c_str();
    hid_t datatype = H5Tcopy (H5T_C_S1);
    H5Tset_size (datatype, H5T_VARIABLE);
    hsize_t str_dimsf[1] { 1 };
    H5::DataSpace dataspace (1, str_dimsf);
    H5::DataSet dset = f->createDataSet(dset_path, datatype, dataspace);

    dset.write(&s, datatype);
}

template <class T>
void H5Writer<T>::write_nvector(std::string dset_path, NVector data) {
    int n = data.size();
    std::vector<std::string> v_real = vec_to_rstring<T>(data);
    std::vector<std::string> v_imag = vec_to_istring<T>(data);

    hid_t datatype = H5Tcopy (H5T_C_S1);
    H5Tset_size (datatype, H5T_VARIABLE);
    std::vector<const char*> s_real (n, nullptr);
    std::vector<const char*> s_imag (n, nullptr);
    for (int i = 0; i < n; i++) {
        s_real[i] = v_real[i].c_str();
        s_imag[i] = v_imag[i].c_str();
    }

    hsize_t str_dimsf[1] { data.size() };
    H5::DataSpace dataspace (1, str_dimsf);
    H5::DataSet real_dset = f->createDataSet(dset_path + "_real", datatype, dataspace);
    H5::DataSet imag_dset = f->createDataSet(dset_path + "_imag", datatype, dataspace);

    real_dset.write(s_real.data(), datatype);
    imag_dset.write(s_imag.data(), datatype);
}

// ************************************************************ //
// ****************** Static template methods ***************** //
// ************************************************************ //
// Note that static template functions must be implemented in header.

/**
 * @brief Mobius transformation C^+ -> D of a list of input data.
 * 
 * @tparam T Precision type to use. 
 * @param z Input vector for mobius transformation. Each element should be in the upper half plane.
 * @return ImaginaryDomainData<T>::NVector Mobius transformed vector. Each element should be in the unit disk.
 */
template <class T>
typename Nevanlinna<T>::NVector Nevanlinna<T>::mobius(const NVector& z) {
    NVector hz(z);
    for (int i = 0; i < z.size(); i++) {
        hz[i] = (z[i] - Prec<T>::I) / (z[i] + Prec<T>::I);
    }
    return hz;
}

/**
 * @brief Inverse mobius transformation D -> C^+.
 * 
 * @tparam T Precision type to use. 
 * @param z Input vector for inverse transformation. Each element should be in the unit disk.
 * @return ImaginaryDomainData<T>::NVector Inverse transformed vector. Each element should be in the upper half plane.
 */
template <class T>
typename Nevanlinna<T>::NVector Nevanlinna<T>::inv_mobius(const NVector& z) {
    NVector hinvz(z);
    for (int i = 0; i < z.size(); i++) {
        hinvz[i] = Prec<T>::I * (Prec<T>::ONE + z[i]) / (Prec<T>::ONE - z[i]);
    }
    return hinvz;
}

/**
 * @brief Computes the Pick matrix from a given set of Y-values (mobius-transformed frequencies) and 
 * lambda-values (mobius-transformed correlators). 
 * 
 * @tparam T Base data type.
 * @param yvals Mobius-transformed Euclidean frequencies.
 * @param lambda_vals Mobius-transformed correlation function data.
 * @return Schur<T>::NMatrix Pick matrix of data.
 */
template <class T> 
typename Schur<T>::NMatrix Schur<T>::get_pick(const NVector& yvals, const NVector& lambda_vals) {
    int N = yvals.size();
    NMatrix pick(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            NComplex num = Prec<T>::ONE - lambda_vals[i] * std::conj(lambda_vals[j]);
            NComplex denom = Prec<T>::ONE - yvals[i] * std::conj(yvals[j]);
            pick(i, j) = num / denom;
        }
    }
    return pick;
}

/**
 * @brief Computes the Pick matrix from a given set of Euclidean frequences (should be 
 * in C^+) and correlation function data (should be in C^+).
 * 
 * @tparam T Base type of data.
 * @param freqs Euclidean frequencies to compute Pick matrix with.
 * @param ng Correlation function data to compute Pick matrix with.
 * @return Schur<T>::NMatrix Pick matrix of data.
 */
template <class T> 
typename Schur<T>::NMatrix Schur<T>::get_pick_realspace(const NVector& freqs, const NVector& ng) {
    NVector yvals = Nevanlinna<T>::mobius(freqs);
    NVector lambda_vals = Nevanlinna<T>::mobius(ng);
    return Schur<T>::get_pick(yvals, lambda_vals);
}

// Zero function for input to Nevanlinna.
template <class T>
typename Schur<T>::NComplex Schur<T>::zero_fcn(const NComplex& z) {
    return NComplex{0, 0};
}

#endif