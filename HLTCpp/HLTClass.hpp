#include <iostream>
#include <time.h>
#include <math.h>
#include <mpfr.h>
#include <eigen3/unsupported/Eigen/MPRealSupport>
#include <eigen3/Eigen/LU>
#include <mpreal.h>

typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic>  MatrixXmp;
typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> VectorXmp;

class HLT
{
	private:
		const mpfr::mpreal ONE   = mpfr::mpreal("1.0");
		const mpfr::mpreal TWO   = mpfr::mpreal("2.0");
		const mpfr::mpreal HALF  = mpfr::mpreal("0.5");
		const mpfr::mpreal PI    = mpfr::const_pi();
		const mpfr::mpreal SQRT2 = sqrt(TWO);
		const mpfr::mpreal SQRTPI = sqrt(PI);

		VectorXmp 	 data;
		MatrixXmp 	 cov;
		VectorXmp	 t;
		int 	  	 T;
		mpfr::mpreal lambda;
		mpfr::mpreal alpha;
		mpfr::mpreal sigma;
		mpfr::mpreal E0;
		bool 	  	 sym;

		// Dependent on [T, sym]
		int 	  t_max;
		VectorXmp R;

		int 	  generate_t_max();


		// Dependent on [T, sym, E_0]
		MatrixXmp W;

		// Dependent on [E_star]
		VectorXmp 	 r;
		mpfr::mpreal Z;
		VectorXmp 	 f;
		VectorXmp 	 g;





	public:
		HLT(const VectorXmp& in_data,
			const MatrixXmp& in_cov,
			int in_T,
		 	bool in_sym,
		 	const mpfr::mpreal& in_sigma,
		 	const mpfr::mpreal& in_alpha,
		 	const mpfr::mpreal& in_E0,
		 	const mpfr::mpreal& in_lambda);
		void print_num_consts();
		/*
		 * Getters and setters
		 */
		// Getters
		VectorXmp 	 get_data();
		MatrixXmp 	 get_cov();
		VectorXmp 	 get_t();
		int 		 get_T();
		mpfr::mpreal get_lambda();
		mpfr::mpreal get_alpha();
		mpfr::mpreal get_sigma();
		mpfr::mpreal get_E0();
		bool		 get_sym();

		int 		 get_t_max();
		VectorXmp	 get_r();
		VectorXmp	 get_R();

		MatrixXmp	 get_W();

		mpfr::mpreal get_Z();
		VectorXmp	 get_f();
		VectorXmp	 get_g();



		// Setters
		void set_data(const VectorXmp& new_data);
		void set_cov(const MatrixXmp& new_cov);
		void set_t(const VectorXmp& new_t);
		void set_T(int new_T);
		void set_lambda(const mpfr::mpreal& new_lambda);
		void set_alpha(const mpfr::mpreal& new_alpha);
		void set_sigma(const mpfr::mpreal& new_sigma);
		void set_E0(const mpfr::mpreal& new_E0);
		void set_sym(bool new_sym);

		void set_t_max(int new_t_max);
		void set_r(const VectorXmp& new_r);
		void set_R(const VectorXmp& new_R);

		void set_W(const MatrixXmp& new_W);

		void set_Z(const mpfr::mpreal& new_Z);
		void set_f(const VectorXmp& new_f);
		void set_g(const VectorXmp& new_g);

		MatrixXmp A();
		MatrixXmp generate_W();

		VectorXmp generate_R();

		VectorXmp 	 generate_r(const mpfr::mpreal& E_star);
		mpfr::mpreal generate_Z(const mpfr::mpreal& E_star);
		mpfr::mpreal F(int k, const mpfr::mpreal& E_star);
		mpfr::mpreal N(int k, const mpfr::mpreal& E_star);
		VectorXmp 	 generate_f(const mpfr::mpreal& E_star);
		VectorXmp 	 generate_g(const mpfr::mpreal& E_star);

		void generate_vars(const mpfr::mpreal& E_star);

		// Used for error analysis
		mpfr::mpreal target(const mpfr::mpreal& E, const mpfr::mpreal& E_star);
		VectorXmp 	 target(const VectorXmp& Es, const mpfr::mpreal& E_star);
		mpfr::mpreal delta_bar(const mpfr::mpreal& E, const mpfr::mpreal& E_star);
		VectorXmp 	 delta_bar(const VectorXmp& Es, const mpfr::mpreal& E_star);
		mpfr::mpreal relative_deviation(const mpfr::mpreal& E, const mpfr::mpreal& in_E_star);
		VectorXmp 	 relative_deviation(const VectorXmp& Es, const mpfr::mpreal& in_E_star);

		// Functionals; useful for optimizing lambda and error analysis
		mpfr::mpreal A_functional(const mpfr::mpreal& in_E_star);
		mpfr::mpreal B_functional();
		mpfr::mpreal W_functional(const mpfr::mpreal& in_E_star, const mpfr::mpreal& in_lambda);
		VectorXmp scan_lambda(const mpfr::mpreal& in_E_star, const VectorXmp& in_lambdas);

		mpfr::mpreal reconstruct();
		MatrixXmp solve(const VectorXmp& Es);




};
