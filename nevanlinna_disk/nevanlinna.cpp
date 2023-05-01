#include <iostream>
#include <omp.h>
#include <chrono>

#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

#include "nevanlinna.hpp"

int main(int argc, char const *argv[]) {

    // Initialize precision
    mpf_set_default_prec(PRECISION);
    std::cout.precision(DIGITS);
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(DIGITS));

    std::cout << std::endl << "Running Nevanlinna." << std::endl;

    // assertm (argc == 4, "Requires 3 arguments: input file name; output file name; eta value.");
    assertm (argc == 5, "Usage : $ ./Nevanlinna <input> <output> <eta> <boson/fermion>");
    std::string in_name = argv[1];
    std::string out_name = argv[2];
    std::string eta_str = argv[3];
    std::string boson_fermion = argv[4];
    bool is_fermion;
    if (boson_fermion.compare("fermion") == 0){
        is_fermion = true;
    }
    else if (boson_fermion.compare("boson") == 0){
        is_fermion = false;
    }
    else {
        throw std::invalid_argument("Please specify 'boson' or 'fermion'.");
    }

    std::cout << "Reading input from: " << in_name << std::endl;
    std::cout << "Writing output to: " << out_name << std::endl;
    std::cout << "Running reconstruction with eta = " << eta_str << std::endl;

    // Read data from input file
    H5Reader<mpfr::mpreal> reader (in_name);
    Prec<mpfr::mpreal>::NVector freqs = reader.get_freqs();         // Matsubara frequencies i\omega_n
    Prec<mpfr::mpreal>::NVector ng = reader.get_ng();               // Green's function data -G(i\omega_n)
    // int beta = freqs.size();
    int beta = reader.get_beta();
    std::cout << "Number of measured Matsubara frequencies: " << freqs.size() << std::endl;
    std::cout << "Beta is: " << beta << std::endl;

    // std::cout << std::endl << "Matsubara frequencies:" << std::endl;
    // print_vector<mpfr::mpreal>(freqs);
    // std::cout << std::endl << "Green's function at Matsubara frequencies:" << std::endl;
    // print_vector<mpfr::mpreal>(ng);

    Prec<mpfr::mpreal>::NReal eta (eta_str);
    Nevanlinna<mpfr::mpreal> nevanlinna (freqs, ng);

    // TODO figuring out problem with NANs
    // std::cout << std::endl << "Phi vals: " << std::endl;
    // print_vector<mpfr::mpreal>(nevanlinna.get_schur().get_w());

    RealDomainData<mpfr::mpreal> omegas;
    Prec<mpfr::mpreal>::NVector rho_recon_disk;
    Prec<mpfr::mpreal>::NVector rho_recon;
    // double start = -1.0;
    // double stop = 1.0;
    double start = 0.0;
    double stop = 3.0;
    int num = 1000;    // Number of points for recon
    std::tie(omegas, rho_recon_disk) = nevanlinna.evaluate(start, stop, num, eta);
    if (is_fermion) {
        std::cout << "Mapping fermionic Green function back to upper half plane." << std::endl;
        rho_recon = Nevanlinna<mpfr::mpreal>::inv_mobius(rho_recon_disk);
    }
    else {
        std::cout << "Mapping bosonic Green function back to upper half plane." << std::endl;
        rho_recon = Nevanlinna<mpfr::mpreal>::inv_ctilde(rho_recon_disk);
    }

    // Compute central value and Wertevorrat
    // std::vector<T> rho_alt;
    // std::vector<T> delta_rho_plus;
    // Prec<mpfr::mpreal>::NVector delta_rho_minus;
    std::cout << "Computing Wertevorrat." << std::endl;
    Prec<mpfr::mpreal>::NVector rho_alt;
    Prec<mpfr::mpreal>::NVector delta_rho_plus;
    Prec<mpfr::mpreal>::NVector delta_rho_minus;
    std::tie(rho_alt, delta_rho_plus, delta_rho_minus) = nevanlinna.wertevorrat(is_fermion);


    // std::cout << std::endl << "Reconstructed frequencies:" << std::endl;
    // print_vector<mpfr::mpreal>(omegas.get_freqs());
    // std::cout << std::endl << "Reconstructed spectral function:" << std::endl;
    // print_vector<mpfr::mpreal>(rho_recon);
    std::cout << "Spectral function reconstructed." << std::endl;

    // Write to output file
    H5Writer<mpfr::mpreal> fout (out_name, beta, start, stop, num, eta, freqs, ng, rho_recon, nevanlinna, rho_alt, delta_rho_plus, delta_rho_minus);
    fout.write();
    std::cout << std::endl << "Recon written to: " << out_name << std::endl;

    return 0;
}