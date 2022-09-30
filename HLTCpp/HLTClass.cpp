// clang++ -lmpfr -lgmp -std=c++11 -fopenmp -o HLTClass HLTClass.cpp
#include <iostream>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Cholesky>
#include "HLTClass.hpp"
#include <omp.h>
#include <chrono>

#define NUM_THREADS 4

HLT::HLT(const VectorXmp& in_data,
		 const MatrixXmp& in_cov,
		 int in_T,
		 bool in_sym,
		 const mpfr::mpreal& in_sigma,
		 const mpfr::mpreal& in_alpha,
		 const mpfr::mpreal& in_E0,
		 const mpfr::mpreal& in_lambda)
{
	/*
	VectorXmp 	 data;
	MatrixXmp 	 cov;
	VectorXmp	 t;
	int 	  	 T = in_T;
	mpfr::mpreal lambda;
	mpfr::mpreal alpha;
	mpfr::mpreal sigma;
	mpfr::mpreal E0;
	bool 	  	 sym;
	*/
	data   = in_data;
	cov    = in_cov;
	T 	   = in_T;
	sym    = in_sym;
	sigma  = in_sigma;
	alpha  = in_alpha;
	E0 	   = in_E0;
	lambda = in_lambda;

	t_max = HLT::generate_t_max();
	R     = HLT::generate_R();
	W 	  = HLT::generate_W();
	std::cout << "Hello!\n";
}

void HLT::print_num_consts()
{
	std::cout << "ONE:  " << ONE   << std::endl;
	std::cout << "TWO:  " << TWO   << std::endl;
	std::cout << "HALF: " << HALF  << std::endl;
	std::cout << "PI:   " << PI    << std::endl;
	std::cout << "SQRT2:" << SQRT2 << std::endl;
}

/*
 * Getters and setters
 */

// Getters
VectorXmp HLT::get_data()
{
	return data;
}

MatrixXmp HLT::get_cov()
{
	return cov;
}

VectorXmp HLT::get_t()
{
	return t;
}

int HLT::get_T()
{
	return T;
}

mpfr::mpreal HLT::get_lambda()
{
	return lambda;
}

mpfr::mpreal HLT::get_alpha()
{
	return alpha;
}

mpfr::mpreal HLT::get_sigma()
{
	return sigma;
}

mpfr::mpreal HLT::get_E0()
{
	return E0;
}

bool HLT::get_sym()
{
	return sym;
}

int HLT::get_t_max()
{
	return t_max;
}

VectorXmp HLT::get_r()
{
	return r;
}

VectorXmp HLT::get_R()
{
	return R;
}

MatrixXmp HLT::get_W()
{
	return W;
}

mpfr::mpreal HLT::get_Z()
{
	return Z;
}

VectorXmp HLT::get_f()
{
	return f;
}

VectorXmp HLT::get_g()
{
	return g;
}


// Setters

void HLT::set_data(const VectorXmp& new_data)
{
	data = new_data;
}

void HLT::set_cov(const MatrixXmp& new_cov)
{
	cov = new_cov;
}

void HLT::set_t(const VectorXmp& new_t)
{
	t = new_t;
}

void HLT::set_T(int new_T)
{
	T = new_T;
}

void HLT::set_lambda(const mpfr::mpreal& new_lambda)
{
	lambda = new_lambda;
}

void HLT::set_alpha(const mpfr::mpreal& new_alpha)
{
	alpha = new_alpha;
}

void HLT::set_sigma(const mpfr::mpreal& new_sigma)
{
	sigma = new_sigma;
}

void HLT::set_E0(const mpfr::mpreal& new_E0)
{
	E0 = new_E0;
}

void HLT::set_sym(bool new_sym)
{
	sym = new_sym;
}

void HLT::set_t_max(int new_t_max)
{
	t_max = new_t_max;
}

void HLT::set_r(const VectorXmp& new_r)
{
	r = new_r;
}

void HLT::set_R(const VectorXmp& new_R)
{
	R = new_R;
}

void HLT::set_W(const MatrixXmp& new_W)
{
	W = new_W;
}

void HLT::set_Z(const mpfr::mpreal& new_Z)
{
	Z = new_Z;
}

void HLT::set_f(const VectorXmp& new_f)
{
	f = new_f;
}

void HLT::set_g(const VectorXmp& new_g)
{
	g = new_g;
}


/*
 *
 * Generated variables and helper functions
 * Variables that only depend on the input parameters: t_max, R, A, and W
 *
 */

int HLT::generate_t_max()
{
	int T = HLT::get_T();
	bool sym = HLT::get_sym();

	if (sym)
	{
		return T / 2;
	}
	return T - 1;
}

VectorXmp HLT::generate_R()
{
	int T = HLT::get_T();
	bool sym = HLT::get_sym();
	int t_max = HLT::get_t_max();

	VectorXmp out = VectorXmp::Zero(t_max);
	for (int t = 0; t < t_max; ++t)
	{
		out(t) = ONE / (t + ONE);
		if (sym)
		{
			out(t) += ONE / (T - t - ONE);
		}
	}
	return out;
}

MatrixXmp HLT::A()
{
	int t_max = HLT::get_t_max();
	int T = HLT::get_T();
	mpfr::mpreal alpha = HLT::get_alpha();
	mpfr::mpreal E0 = HLT::get_E0();

	MatrixXmp out = MatrixXmp::Zero(t_max, t_max);

	mpfr::mpreal denom1;
	mpfr::mpreal denom2;
	mpfr::mpreal denom3;
	mpfr::mpreal denom4;

	for (int i = 0; i < t_max; ++i)
	{
		for (int j = 0; j < t_max; ++j)
		{
			denom1 = ( i +  j + TWO - alpha);

			out(i, j) = exp(-E0 * denom1) / denom1;
			/* Need to add PBC terms
			if (sym)
			{
				denom2 = ( T - i +  j - alpha);
				denom3 = ( T + i -  j - alpha);
				denom4 = ( TWO * T - i -  j - TWO - alpha);

				out(i, j) += exp(-E0 * denom2) / denom2;
				out(i, j) += exp(-E0 * denom3) / denom3;
				out(i, j) += exp(-E0 * denom4) / denom4;
			}
			*/
		}
	}
	return out;
}

// Need to fully implement, for test purposes W is just the A matrix
MatrixXmp HLT::generate_W()
{
	int 		 t_max = HLT::get_t_max();
	mpfr::mpreal lambda = HLT::get_lambda();
	MatrixXmp 	 A = HLT::A();
	MatrixXmp 	 cov = HLT::get_cov();
	mpfr::mpreal data0 = mpfr::mpreal(HLT::get_data()(0));

	if(lambda < std::numeric_limits<mpfr::mpreal>::epsilon()){
		return A;
	}

	MatrixXmp out = MatrixXmp::Zero(t_max, t_max);
	for (size_t i = 0; i < t_max; i++) {
		for (size_t j = 0; j < t_max; j++) {
			out(i, j) = (ONE - lambda) * A(i, j) + (lambda/(data0 * data0)) * cov(i, j);
		}
	}

	return out;
}

/*
 *
 * Energy-dependent generated variables and helper functions
 * Variables that depend on a given E_star: r, Z, F, N, f, g, and generate_vars
 *
 */

VectorXmp HLT::generate_r(const mpfr::mpreal& in_E_star)
{
	int T = HLT::get_T();
	bool sym = HLT::get_sym();
	int t_max = HLT::get_t_max();

	VectorXmp out = VectorXmp::Zero(T);
	for (int t = 0; t < T; ++t)
	{
		out(t) = exp(-in_E_star * t);
		if (sym)
		{
			out(t) += exp(-in_E_star * (T - t));
		}
	}
	return out;
}

mpfr::mpreal HLT::generate_Z(const mpfr::mpreal& in_E_star)
{
	mpfr::mpreal sigma = HLT::get_sigma();
	return HALF * (ONE + mpfr::erf(in_E_star / (sigma * SQRT2)));
}

mpfr::mpreal HLT::F(int k, const mpfr::mpreal& in_E_star)
{
	mpfr::mpreal sigma = HLT::get_sigma();
	mpfr::mpreal alpha = HLT::get_alpha();
	mpfr::mpreal E0    = HLT::get_E0();

	mpfr::mpreal F_arg = ((alpha - k) * sigma * sigma + in_E_star - E0) / (SQRT2 * sigma);
	return ONE + mpfr::erf(F_arg);
}

mpfr::mpreal HLT::N(int k, const mpfr::mpreal& in_E_star)
{
	mpfr::mpreal sigma  = HLT::get_sigma();
	mpfr::mpreal alpha  = HLT::get_alpha();
	mpfr::mpreal lambda = HLT::get_lambda();

	mpfr::mpreal N_arg = HALF * ((alpha - k) * ((alpha - k) * sigma * sigma + TWO * in_E_star) );
	return (ONE - lambda) / (TWO * HLT::generate_Z(in_E_star)) * mpfr::exp(N_arg);
}

VectorXmp HLT::generate_f(const mpfr::mpreal& in_E_star)
{
	int T = HLT::get_T();
	int t_max = HLT::get_t_max();
	bool sym = HLT::get_sym();

	VectorXmp out = VectorXmp::Zero(t_max);
	for (int t = 0; t < t_max; ++t)
	{
		out(t) = HLT::F(t + 1, in_E_star) * HLT::N(t + 1, in_E_star);
		if (sym){
			out(t) += HLT::F(T - t - 1, in_E_star) * HLT::N(T - t - 1, in_E_star);
		}
	}
	return out;
}

VectorXmp HLT::generate_g(const mpfr::mpreal& in_E_star)
{
	MatrixXmp W = HLT::get_W();
	VectorXmp R = HLT::get_R();
	VectorXmp f = HLT::get_f();

	VectorXmp Winv_f;
	VectorXmp Winv_R;

	/*
	Eigen::FullPivHouseholderQR<MatrixXmp> qr(W);
	Winv_R = qr.solve(R);
    Winv_f = qr.solve(f);
	*/


	Eigen::LDLT<MatrixXmp> ldlt(W);
	Winv_R = ldlt.solve(R);
	Winv_f = ldlt.solve(f);


	/*
	Eigen::JacobiSVD<MatrixXmp> svd(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Winv_R = svd.solve(R);
    Winv_f = svd.solve(f);
	*/

	/*
	Eigen::BDCSVD<MatrixXmp> bdcsvd(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Winv_R = bdcsvd.solve(R);
    Winv_f = bdcsvd.solve(f);
	*/


	Eigen::FullPivLU<MatrixXmp> lu(W);
	Winv_R = lu.solve(R);
    Winv_f = lu.solve(f);


	/*
	std::cout << "R       :\n" << R << std::endl;
	std::cout << "SVD R	  :\n" << (W * Winv_R - R)<<  std::endl;
	std::cout << "LU R	  :\n" << (W * lu.solve(R) - R) << std::endl;
	*/

	//std::cout << "SVD relative error:\n" << (W * Winv_R - R).norm() / R.norm() <<  std::endl;
	//std::cout << "LU relative error:\n" << (W * lu.solve(R) - R).norm() / R.norm() << std::endl;
	//std::cout << "QR relative error:\n" << (W * qr.solve(R) - R).norm() / R.norm() << std::endl;
	//std::cout << "LDLT relative error:\n" << (W * ldlt.solve(R) - R).norm() / R.norm() << std::endl;
	VectorXmp out = Winv_f + Winv_R * (ONE - R.transpose() * Winv_f) / (R.transpose() * Winv_R);

	return out;
}

void HLT::generate_vars(const mpfr::mpreal& in_E_star)
{
	HLT::set_r(HLT::generate_r(in_E_star));
	HLT::set_Z(HLT::generate_Z(in_E_star));
	HLT::set_f(HLT::generate_f(in_E_star));
	HLT::set_g(HLT::generate_g(in_E_star));
}

mpfr::mpreal HLT::target(const mpfr::mpreal& E, const mpfr::mpreal& in_E_star)
{
	mpfr::mpreal sigma = HLT::get_sigma();
	mpfr::mpreal Z = HLT::generate_Z(in_E_star);

	mpfr::mpreal denom = ONE / (sqrt(TWO * PI) * sigma * Z);

	return denom * exp(-ONE * pow((E - in_E_star), 2) / (TWO * pow(sigma, 2)));
}

VectorXmp HLT::target(const VectorXmp& Es, const mpfr::mpreal& in_E_star)
{
	/*
	mpfr::mpreal sigma = HLT::get_sigma();
	mpfr::mpreal Z = HLT::generate_Z(E_star); //Might need to change get to generate in full implementation

	mpfr::mpreal denom = ONE / (sqrt(TWO * PI) * sigma * Z);
	*/
	VectorXmp out = VectorXmp::Zero(Es.rows());
	for (int i = 0; i < Es.rows(); i++) {
		//out(i) = denom * exp(-ONE * pow((Es(i) - E_star), 2) / (TWO * pow(sigma, 2)));
		out(i) = HLT::target(Es(i), in_E_star);
	}

	return out;
}

mpfr::mpreal HLT::delta_bar(const mpfr::mpreal& E, const mpfr::mpreal& in_E_star)
{
	VectorXmp g = HLT::generate_g(in_E_star); //Might need to change get to generate in full implementation
	VectorXmp r = HLT::generate_r(E);  	   //Might need to change get to generate in full implementation
	mpfr::mpreal out = mpfr::mpreal("0");
	for (int i = 0; i < g.rows(); i++) {
		out += g(i) * r(i + 1);
	}
	return out;
}

VectorXmp HLT::delta_bar(const VectorXmp& Es, const mpfr::mpreal& in_E_star)
{
	HLT::generate_vars(in_E_star);
	VectorXmp out = VectorXmp::Zero(Es.rows());
	for (size_t i = 0; i < Es.rows(); i++) {
		out(i) = HLT::delta_bar(Es(i), in_E_star);
	}

	return out;
}

mpfr::mpreal HLT::relative_deviation(const mpfr::mpreal& E, const mpfr::mpreal& in_E_star)
{
	mpfr::mpreal dbar = HLT::delta_bar(E, in_E_star);
	mpfr::mpreal targ = HLT::target(E, in_E_star);

	return ONE - dbar / targ;
}

VectorXmp HLT::relative_deviation(const VectorXmp& Es, const mpfr::mpreal& in_E_star)
{
	HLT::generate_vars(in_E_star);
	VectorXmp out = VectorXmp::Zero(Es.rows());
	for (size_t i = 0; i < Es.rows(); i++) {
		out(i) = HLT::relative_deviation(Es(i), in_E_star);
	}

	return out;
}

mpfr::mpreal HLT::A_functional()
{
	mpfr::mpreal lambda = HLT::get_lambda();
	return NULL;
}

mpfr::mpreal HLT::B_functional()
{
	mpfr::mpreal lambda = HLT::get_lambda();
	VectorXmp cov		= HLT::get_cov();
	VectorXmp g 		= HLT::get_g();

	return lambda * (g.transpose() * cov * g)(0);
}

mpfr::mpreal HLT::reconstruct()
{
	int t_max 	   = HLT::get_t_max();
	VectorXmp data = HLT::get_data();
	VectorXmp g	   = HLT::get_g();



	mpfr::mpreal out = mpfr::mpreal("0.0");
	for (int i = 0; i < t_max; i++) {
		out += g(i) * data(i + 1);
	}

	return out;
}

MatrixXmp HLT::solve(const VectorXmp& Es)
{
	auto start = std::chrono::high_resolution_clock::now();
	std::cout << "pre-out" << '\n';
	MatrixXmp out = MatrixXmp::Zero(2, Es.rows());
	std::cout << "post-out" << '\n';
	//#pragma omp parallel for
	for (int i = 0; i < Es.rows(); i++) {
		std::cout << i << " of " << Es.rows() << '\n';
		generate_vars(Es(i));
		std::cout << "pre-reconstruct" << '\n';
		out(0, i) = reconstruct();
		std::cout << "pre-relative_deviation" << '\n';
		out(1, i) = out(0, i) * abs(HLT::relative_deviation(Es(i), Es(i)));
		std::cout << "finish" << '\n';
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken by function: "
         << duration.count() << " microseconds" << std::endl;

	return out;
}


int main(int argc, char const *argv[])
{
	//omp_set_num_threads(NUM_THREADS);
	const int digits = 128;
	mpfr::mpreal::set_default_prec(mpfr::digits2bits(digits));
	std::cout.precision(digits);
	int T = 65;
	bool sym = 0;
	mpfr::mpreal sigma  = mpfr::mpreal("0.05");
	mpfr::mpreal alpha  = mpfr::mpreal("0.00");
	mpfr::mpreal E0     = mpfr::mpreal("0.0");
	mpfr::mpreal lambda = mpfr::mpreal("0.01");
	mpfr::mpreal E_star = mpfr::mpreal("0.5");
	int t = 1;
	VectorXmp data = VectorXmp::Zero(T);
	for (size_t i = 0; i < data.rows(); i++)
	{
		data(i) = exp(-mpfr::mpreal("0.2") * i) + exp(-mpfr::mpreal("0.5") * i) + exp(-mpfr::mpreal("0.8") * i);
	}

	MatrixXmp cov = MatrixXmp::Zero(T, T);
	for (size_t i = 0; i < cov.rows(); i++)
	{
		for (size_t j = 0; j < cov.cols(); j++)
		{
			cov(i, j) = mpfr::mpreal("0.01");// * data(i) * data(j);
		}
	}

	//std::cout << data << std::endl;
/*
	const mpfr::mpreal E_star = mpfr::mpreal("0.1");
	const mpfr::mpreal sigma = mpfr::mpreal("0.05");
	const mpfr::mpreal alpha = mpfr::mpreal("0.015");
	const mpfr::mpreal lambda = mpfr::mpreal("0.3");
	const mpfr::mpreal E_0	  = mpfr::mpreal("0.2");
*/

	VectorXmp Es = VectorXmp::LinSpaced(101, 0, 1);

	HLT test = HLT(data, cov, data.rows(), sym, sigma, alpha, E0, lambda);
	test.generate_vars(mpfr::mpreal("1e-10"));
	//test.print_num_consts();

	//std::cout << "R:\n" <<  test.generate_R().transpose() << "\n" << std::endl;
	/*
	std::cout << "A:\n" <<  test.A() << "\n" << std::endl;
	std::cout << "Z(" << E_star << "): " << test.generate_Z(E_star) << std::endl;
	std::cout << "F(" << E_star << ", " << t << "): " << test.F(t, E_star) << std::endl;
	std::cout << "N(" << E_star << ", " << t << "): " << test.N(t, E_star) << std::endl;
	std::cout << "f(" << E_star << "): " << test.generate_f(E_star) << std::endl;
	*/
	//std::cout << "g(" << E_star << "):\n"<< test.generate_g(E_star) << std::endl;
	//std::cout << "target(Es, " << E_star << "):\n"<< test.target(Es, E_star) << std::endl;

	//VectorXmp temp  = test.delta_bar(Es, mpfr::mpreal("0.2"));
	//VectorXmp temp2 = test.target(Es, mpfr::mpreal("0.5"));
	//VectorXmp temp3 = test.target(Es, mpfr::mpreal("0.8"));test.target(Es, 0.2) + test.target(Es, 0.5) + test.target(Es, 0.8);
	MatrixXmp temp4 = test.solve(Es);
	//VectorXmp temp5 = test.get_g();
	for (size_t i = 0; i < temp4.cols(); i++) {
		//std::cout << temp(i) + temp2(i) + temp3(i) << ", ";
		std::cout << temp4(1, i) << ", ";
	}

	/*
	 * Try to transform back to the correlator data
	 */


	return 0;
}
