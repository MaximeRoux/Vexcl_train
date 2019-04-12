#include <stdio.h>
#include <stdexcept>
#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


#define VEXCL_BACKEND_CUDA

#include <vexcl/vexcl.hpp>
#include <vexcl/vector_view.hpp>

#ifdef LSWM_WITH_VEXCL
using RealType = float;
//vex::Context ctx(vex::Filter::GPU);;
typedef vex::vector<RealType> SpatialBlockField1D;
#else
typedef Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> SpatialBlockField1D;
#endif



namespace L2S
{
	enum Directions {
		X = 0,
		Y,
		Z,
		DIM
	};

	enum StressFieldComponents {
		XX = 0,
		YY,
		ZZ,
		XY,
		XZ,
		YZ,
		NB_STRESS_FIELD_COMPONENTS
	};

	enum Locations {
		LEFT = 0,
		RIGHT,
		BACKWARD,
		FORWARD,
		BOTTOM,
		TOP,
		NB_LOCATIONS
	};


	enum DIJK { DI = 0, DJ = 0, DK = 0 }; // Identifiers to be used when applying the finite-difference operator

	enum ClusterNodes { MP_0, MP_1, MP_2, MP_3, MP_4, MP_5, MP_6, MP_7, MP_8, MP_9, MP_10, MP_11, MP_12, MP_13, MP_14, MP_15, MP_16, MP_17, NB_CLUSTER_NODES }; // Cluster nodes
	enum Cores { C0, C1, C2, C3, NB_CORES }; // CPU cores

	enum ColorizationStrategies { CORE, TIME_STEP };
}





/*--------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
-------------------------------------------------EIGEN----------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
*/



namespace L2S
{

	template<typename RealType>
	class SpatialBlockField
	{

	public:
		SpatialBlockField() :nx_(1),
			ny_(1),
			nz_(1)
		{
			// Add padding along z-axis for vectorization efficiency
			px_ = 0;
			py_ = 0;
			pz_ = std::ceil(RealType(nz_) / packetSize_)*packetSize_ - nz_;

			nz_ += pz_;

			n_ = nx_ * ny_*nz_;

			data_.resize(nx_, ny_);

			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {

#ifdef LSWM_WITH_VEXCL
					auto & dij = data_(i, j);
					dij.resize(nz_);
					std::cout << " vector vexcl size(data(i,j)) = " << data_(i, j).size() << " && nz_ =" << nz_ << std::endl;
#else 
					auto & dij = data_(i, j);
					dij.resize(1, nz_);
#endif
				}
			}
		}

		SpatialBlockField(const int nx, const int ny, const int nz) :nx_(nx),
			ny_(ny),
			nz_(nz)
		{
			// Add padding along z-axis for vectorization efficiency
			std::cout << " AVANT MODIF nx_ = " << nx_ << " ny_ = " << ny_ << " && nz_ =" << nz_ << std::endl;
			px_ = 0;
			py_ = 0;
			pz_ = std::ceil(RealType(nz_) / packetSize_)*packetSize_ - nz_;

			nz_ += pz_; 

			std::cout << " APRES MODIF nx_ = " << nx_ << " ny_ = " << ny_ << " && nz_ =" << nz_ << std::endl;
			
			n_ = nx_ * ny_*nz_;

			data_.resize(nx_, ny_);
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					
#ifdef LSWM_WITH_VEXCL
					auto & dij = data_(i, j);
					dij.resize( nz_);
					std::cout << " vector vexcl size(data(i,j)) = " << data_(i, j).size() << " && nz_ =" << nz_ << std::endl;
#else 
					auto & dij = data_(i, j);
					dij.resize(1, nz_);
#endif
				}
			}
		}

		SpatialBlockField(const SpatialBlockField & o)
		{
			// std::cout << "Building SpatialBlockField from SpatialBlockField\n";
			if (this != &o) {
				nx_ = o.nx_;
				ny_ = o.ny_;
				nz_ = o.nz_;
				n_ = o.n_;

				// hnx_=o.hnx_;
				// hny_=o.hny_;
				// hnz_=o.hnz_;

				px_ = o.px_;
				py_ = o.py_;
				pz_ = o.pz_;

				// FIXME we just need to copy the data without modifying source locations
				// hasSource_=o.hasSource_;

				// is_=o.is_;
				// js_=o.js_;
				// ks_=o.ks_;

				data_ = o.data_;
			}
		}

		// template <typename E>
		// SpatialBlockField(const SpatialBlockFieldExpression<E> & e):nx_(e.nx()), ny_(e.ny()), nz_(e.nz()),
		//                                                             n_(e.n()),
		//                                                             px_(e.px()), py_(e.py()), pz_(e.pz())
		// {
		//   data_.resize(nx_, ny_);
		//   for (int i=0; i<nx_; i++){
		//     for (int j=0; j<ny_; j++){
		//       data_(i,j).resize(1,nz_);
		//       data_(i,j)=e.get(i,j);
		//     }
		//   }
		// }

		~SpatialBlockField()
		{
		}

		inline auto dimension(const short d) const
		{
			size_t dim;

			switch (d) {
			case X:
				dim = nx_;
				break;
			case Y:
				dim = ny_;
				break;
			case Z:
				dim = nz_;
				break;
			default:
				std::cerr << "SpatialBlockField::dimension() - Unknown dimension: " << d << "\n";
				exit(-1);
			}

			return dim;
		}

		inline auto dimensions() const
		{
			return std::make_tuple<const int, const int, const int>((const int)nx_, (const int)ny_, (const int)nz_);
		}

		inline void setHaloSize(const int hnx, const int hny, const int hnz)
		{
			hnx_ = hnx;
			hny_ = hny;
			hnz_ = hnz;
		}

		inline void resize(const int nx, const int ny, const int nz)
		{
			nx_ = nx;
			ny_ = ny;
			nz_ = nz;

			// Add padding along z-axis for vectorization efficiency
			px_ = 0;
			py_ = 0;
			pz_ = std::ceil(RealType(nz_) / packetSize_)*packetSize_ - nz_;

			nz_ += pz_;

			n_ = nx_ * ny_*nz_;

			data_.resize(nx_, ny_);
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					auto & dij = data_(i, j);
#ifdef LSWM_WITH_VEXCL
					dij.resize(nz_);
#else 
					dij.resize(1, nz_);
#endif
				}
			}
		}

		inline void addSource(const int is, const int js, const int ks)
		{
			hasSource_ = true;

			is_.push_back(is);
			js_.push_back(js);
			ks_.push_back(ks);
		}


		inline const auto & nx() const { return nx_; }
		inline const auto & ny() const { return ny_; }
		inline const auto & nz() const { return nz_; }

		inline const auto & hnx() const { return hnx_; }
		inline const auto & hny() const { return hny_; }
		inline const auto & hnz() const { return hnz_; }

		inline const auto & px() const { return px_; }
		inline const auto & py() const { return py_; }
		inline const auto & pz() const { return pz_; }

		inline const auto & n() const { return n_; }

		inline auto & data() { return data_; }
		inline const auto & data() const { return data_; }

		inline const auto iStart() const { return hnx_; }
		inline const auto iEnd() const { return nx_ - hnx_ - px_; }

		inline const auto jStart() const { return hny_; }
		inline const auto jEnd() const { return ny_ - hny_ - py_; }

		inline const auto kStart() const { return hnz_; }
		inline const auto kEnd() const { return nz_ - hnz_ - pz_; }

		inline auto index(const int i, const int j, const int k) const { return k * nx_*ny_ + j * nx_ + i; }

		inline const auto & hasSource() const { return hasSource_; }

		inline const auto & is() const { return is_; }
		inline const auto & js() const { return js_; }
		inline const auto & ks() const { return ks_; }

		inline auto & operator=(const SpatialBlockField && o)
		{
			// std::cout << "Building SpatialBlockField from SpatialBlockField through move semantics\n";
			if (this != &o) {
				nx_ = o.nx_;
				ny_ = o.ny_;
				nz_ = o.nz_;

				n_ = o.n_;

				hnx_ = o.hnx_;
				hny_ = o.hny_;
				hnz_ = o.hnz_;

				px_ = o.px_;
				py_ = o.py_;
				pz_ = o.pz_;

				// FIXME we just need to copy the data without modifying source locations
				// hasSource_=o.hasSource_;

				// is_=o.is_;
				// js_=o.js_;
				// ks_=o.ks_;

				data_ = o.data_;
			}
			return *this;
		}

		inline auto & operator=(const RealType & v)
		{
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					data_(i, j) = v;
				}
			}
			return *this;
		}



		inline const auto & get(const int i, const int j) const
		{
			return data_(i, j);
		}

		



		inline auto & operator()(const int i, const int j)
		{
			return data_(i, j);
		}

		inline const auto & operator()(const int i, const int j) const
		{
			return data_(i, j);
		}
#ifdef LSWM_WITH_VEXCL
		inline auto & operator()(const int i, const int j, const int k)
		{
			/*std::cout << " vector vexcl size(data(i,j)) = " << data_(i, j).size() << std::endl;
			auto & dij = data_(i, j);
			std::cout << " vector vexcl size(dij) = " << dij.size() << std::endl;
			std::cout << " vector vexcl size data = " << data_.size() << std::endl;*/
			return data_(i, j)[k];
		}

		inline const auto & operator()(const int i, const int j, const int k) const
		{
			return data_(i, j)[k];
		}

		inline const auto & get(const int i, const int j, const int k) const
		{
			return data_(i, j)[k];
		}

		inline void display() const
		{
			for (int k = kStart(); k < kEnd(); k++) {
				std::cerr << "k = " << k - hnz_ << " *********************************" << std::endl;
				for (int i = iStart(); i < iEnd(); i++) {
					for (int j = jStart(); j < jEnd(); j++) {
						std::cerr << data_(i, j)[k] << " ";
		}
					std::cerr << std::endl;
				}
			}
		}
#else
		inline auto & operator()(const int i, const int j, const int k)
		{
			return data_(i, j)(k);
		}

		inline const auto & operator()(const int i, const int j, const int k) const
		{
			return data_(i, j)(k);
		}

		inline const auto & get(const int i, const int j, const int k) const
		{
			return data_(i, j)(k);
		}

		inline void display() const
		{
			for (int k = kStart(); k < kEnd(); k++) {
				std::cerr << "k = " << k - hnz_ << " *********************************" << std::endl;
				for (int i = iStart(); i < iEnd(); i++) {
					for (int j = jStart(); j < jEnd(); j++) {
						std::cerr << data_(i, j)(k) << " ";
					}
					std::cerr << std::endl;
				}
			}
		}
#endif




		inline auto operator+(const SpatialBlockField & o) const
		{
			SpatialBlockField r(nx_, ny_, nz_);
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					r.data_(i, j) = data_(i, j) + o.data()(i, j);
				}
			}
			return r;
		}

		inline auto & operator+=(const SpatialBlockField & o)
		{
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					data_(i, j) += o.data()(i, j);
				}
			}
			return *this;
		}

		// template<typename E>
		// inline auto & operator+=(const SpatialBlockFieldExpression<E> & e)
		// {
		//   for (int i=0; i<nx_; i++){
		//     for (int j=0; j<ny_; j++){
		//       data_(i,j)+=e.get(i,j);
		//     }
		//   }
		//   return *this;
		// }

		inline auto operator-(const SpatialBlockField & o) const
		{
			SpatialBlockField r(nx_, ny_, nz_);
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					r.data_(i, j) = data_(i, j) - o.data()(i, j);
				}
			}
			return r;
		}

		inline auto & operator-=(const SpatialBlockField & o)
		{
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					data_(i, j) -= o.data()(i, j);
				}
			}
			return *this;
		}

		inline auto operator*(const SpatialBlockField & o) const
		{
			SpatialBlockField r(nx_, ny_, nz_);
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					r.data_(i, j) = data_(i, j)*o.data()(i, j);
				}
			}
			return r;
		}

		inline auto operator*(const RealType & v) const
		{
			SpatialBlockField r(nx_, ny_, nz_);
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					r.data_(i, j) = v * data_(i, j);
				}
			}
			return r;
		}

		inline friend SpatialBlockField operator*(const RealType & v, const SpatialBlockField & o)
		{
			auto nx = o.dimension(X);
			auto ny = o.dimension(Y);
			auto nz = o.dimension(Z);

			SpatialBlockField r(nx, ny, nz);
			for (int i = 0; i < nx; i++) {
				for (int j = 0; j < ny; j++) {
					r.data_(i, j) = v * o.data_(i, j);
				}
			}
			return r;
		}

		inline auto & operator*=(const RealType & v)
		{
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					data_(i, j) *= v;
				}
			}
			return *this;
		}

		inline auto & operator*=(const SpatialBlockField & o)
		{
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					data_(i, j) *= o.data()(i, j);
				}
			}
			return *this;
		}

		inline friend SpatialBlockField operator/(const RealType & v, const SpatialBlockField & o)
		{
			auto nx = o.dimension(X);
			auto ny = o.dimension(Y);
			auto nz = o.dimension(Z);

			SpatialBlockField r(nx, ny, nz);
			for (int i = 0; i < nx; i++) {
				for (int j = 0; j < ny; j++) {
					r.data_(i, j) = v / o.data_(i, j);
				}
			}

			return r;
		}

		inline friend SpatialBlockField operator/(const SpatialBlockField & o, const RealType & v)
		{
			auto nx = o.dimension(X);
			auto ny = o.dimension(Y);
			auto nz = o.dimension(Z);

			SpatialBlockField r(nx, ny, nz);
			for (int i = 0; i < nx; i++) {
				for (int j = 0; j < ny; j++) {
					r.data_(i, j) = o.data_(i, j) / v;
				}
			}

			return r;
		}

		inline auto & operator/=(const RealType & v)
		{
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					data_(i, j) /= v;
				}
			}
			return *this;
		}

		inline auto & operator/=(const SpatialBlockField & o)
		{
			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					data_(i, j) /= o.data()(i, j);
				}
			}
			return *this;
		}

#ifdef BOOST_SIMD_VECTORIZATION
		static inline auto getPack(const SpatialBlockField1D & v1D, const int kp)
		{
			const size_t offset = kp * Eigen::internal::packet_traits<RealType>::size;
			return boost::simd::aligned_load<boost::simd::pack<RealType>>((RealType *)(v1D.data() + offset));
		}

		static inline void setPack(const boost::simd::pack<RealType> & pack, SpatialBlockField1D & v1D, const int kp)
		{
			const size_t offset = kp * Eigen::internal::packet_traits<RealType>::size;
			boost::simd::aligned_store(pack, v1D.data() + offset);
		}
#endif

		

		inline const auto norm2() const
		{
			const auto _iStart = iStart();
			const auto _jStart = jStart();
			const auto _kStart = kStart();

			const auto _iEnd = iEnd();
			const auto _jEnd = jEnd();
			const auto _kEnd = kEnd();

			L2S::RealType n2 = 0.0;

			for (int i = _iStart; i < _iEnd; i++) {
				for (int j = _jStart; j < _jEnd; j++) {
					for (int k = _kStart; k < _kEnd; k++) {
						n2 += get(i, j, k)*get(i, j, k);
					}
				}
			}

			return n2;
		}

		inline void plot1D(const int j0, const int k0,
			const std::string name) const
		{
#ifdef USE_MATPLOTLIB
			std::ofstream of(name + "-" + std::to_string(j0) + "-" + std::to_string(k0) + ".txt");

			namespace mpl = matplotlibcpp;

			const auto n = nx_;
			std::vector<RealType> x(n), y(n);
			for (int i = 0; i < nx_; i++) {
				x.at(i) = i;
				y.at(i) = data_(i, j0)(k0);

				of << int(x.at(i)) << " " << std::scientific << y.at(i) << std::endl;
			}

			of.close();


			mpl::named_plot(name, x, y);

			mpl::xlim(0, nx_);

			mpl::xlabel("i");
			mpl::ylabel(name + "(i," + std::to_string(j0) + "," + std::to_string(k0) + ")");

			mpl::legend();

			mpl::save("./" + name + ".pdf");
#endif
		}

		inline void plot2D(const int k0,
			const std::string name) const
		{
#ifdef USE_MATPLOTLIB
			std::ofstream of(name + "-" + std::to_string(k0) + ".txt");

			for (int i = 0; i < nx_; i++) {
				for (int j = 0; j < ny_; j++) {
					of << i << " " << j << " " << std::scientific << data_(i, j)(k0) << std::endl;
				}
			}

			of.close();
#endif
		}

		inline void plot3D() const
		{
		}

		inline void plotHalo(const L2S::Locations l, const std::string name) const
		{
#ifdef USE_MATPLOTLIB

			switch (l) {
			case LEFT: {
				for (int i = 0; i < hnx_; i++) {
					std::ofstream of(name + "-" + std::to_string(i) + ".txt");

					for (int j = 0; j < ny_; j++) {
						for (int k = 0; k < nz_; k++) {
							of << j << " " << k << " " << std::scientific << data_(i, j)(k) << std::endl;
						}
					}

					of.close();
				}
				break;
			}
			case RIGHT: {
				for (int i = nx_ - hnx_; i < nx_; i++) {
					std::ofstream of(name + "-" + std::to_string(i) + ".txt");

					for (int j = 0; j < ny_; j++) {
						for (int k = 0; k < nz_; k++) {
							of << j << " " << k << " " << std::scientific << data_(i, j)(k) << std::endl;
						}
					}

					of.close();
				}
				break;
			}
			default:
				break;
			}

#endif
		}

	private:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW // To force an object of this class to be allocated as aligned

			enum {
			X = 0,
			Y,
			Z,
			DIM
		};

		const int packetSize_ = Eigen::internal::packet_traits<RealType>::size;

		int nx_;
		int ny_;
		int nz_;

		/* Halo size along x, y and z */
		// FIXME find a better way to define the halo size!!!
		int hnx_ = 2;
		int hny_ = 2;
		int hnz_ = 2;

		/* Padding along x, y and z */
		int px_ = 0;
		int py_ = 0;
		int pz_ = 0;

		int n_;

		/* Indicates whether this spatial block contains a source */
		bool hasSource_ = false;

		/* Coordinates of the sources located within this spatial block */
		std::vector<int> is_;
		std::vector<int> js_;
		std::vector<int> ks_;



		// (i,j)(k)
		Eigen::Array<SpatialBlockField1D, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> data_;

#ifdef LSWM_WITH_VEXCL
		//vex::Context ctx{ vex::Filter::GPU };
#endif // LSWM_WITH_VEXCL


	};

}





void test() {

	size_t size = 11;

	auto coef = 2.4f;
	std::vector<float> a = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f };
	std::vector<float> b = { 2.f, 1.f, 4.f, 5.f, 7.f, 1.f, 8.f, 6.f, 7.f, 1.f, 2.f };
	std::vector<float> c = { 1.f, 3.f, 2.f, 6.f, 5.f, 4.f, 8.f, 10.f, 9.f, 1.f, 2.f };
	std::vector<float> d(size);


	vex::Context ctx(vex::Filter::GPU); //Contexte
	vex::vector<float> a_d( size);
	vex::vector<float> b_d(size), c_d(size), res_d(size);  // Device vector.
	vex::copy(a, a_d);    // Copy data from host to device.
	vex::copy(b, b_d);    // Copy data from host to device.
	vex::copy(c, c_d);    // Copy data from host to device.

	a_d[0] = 1.f;
	res_d = a_d + b_d - 2.4f * c_d; //Compute on device

	vex::copy(res_d, d);// Copy data from device to host.

	for (size_t i = 0; i < d.size(); i++)
	{
		std::cout << "d[" << i << "] = " << d[i] << '\n';
	}

	int n = 10;

	using vex::range;
	using vex::_;
	/*
	//vex::slicer<NDim>
	//vex::vector<float> X(ctx, a);
	vex::vector<double> X(ctx, n * n); // n-by-n matrix stored in row-major order.
	vex::vector<double> Y(ctx, n);
	for (size_t i = 0; i < X.size(); i++)
	{
		X[i] = i;
	}

	// vex::extents is a helper object similar to boost::multi_array::extents.
	vex::slicer<2> slice(vex::extents[n][n]);

	Y = slice[5](X);    // Put 5-th row of X into Y.

	for (size_t i = 0; i < Y.size(); i++)
	{
		std::cout << "Y[" << i << "] = " << Y[i] << '\n';
	}
	Y = slice[_][5](X); // Put 5-th column of X into Y.

	for (size_t i = 0; i < Y.size(); i++)
	{
		std::cout << "Y[" << i << "] = " << Y[i] << '\n';
	}

	slice[0][0](X) = 561; // Put Y into 5-th column of X.


	for (size_t i = 0; i < X.size(); i++)
	{
		if(X[i]!=i)
			std::cout << "!!!!!!! X[" << i << "] = " << X[i] << '\n';
	}
	// Assign sub-block [10,20)x[30,40) of X to Z:
	vex::vector<double> Z = slice[range(10, 20)][range(30, 40)](X);
	assert(Z.size() == 100);
	*/




}



void bench() {

	size_t n = 1000000;
	std::vector<float> a(n), b(n), c(n), d(n); // Host vector.

	vex::slicer<3> slice(vex::extents[n][n][n]);

	vex::Context ctx(vex::Filter::GPU); //Contexte
	vex::vector<float> a_d(n), b_d(n), c_d(n), res_d(n);  // Device vector.
	for (size_t i = 0; i < n; i++)
	{

		a[i] = (i*1.0f + 4.)*0.5;
		b[i] = (i*1.0f + 5.)*0.5;
		c[i] = (i*1.0f + 6.)*0.5;
	}

	vex::copy(a, a_d);    // Copy data from host to device.
	vex::copy(b, b_d);    // Copy data from host to device.
	vex::copy(c, c_d);    // Copy data from host to device.

	for (size_t i = 0; i < n; i++)
	{
		res_d = a_d + b_d - 2.4f * c_d; //Compute on device
	}


	vex::copy(res_d, d);// Copy data from device to host.
	for (size_t i = 0; i < n; i = i + 1000000)
	{
		printf("d[%d] = %f\n", i, d[i]);
	}


}

void test_multi_vector() {
	size_t size = 11;

	auto coef = 2.4f;
	std::vector<float> a = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f };
	std::vector<float> b = { 2.f, 1.f, 4.f, 5.f, 7.f, 1.f, 8.f, 6.f, 7.f, 1.f, 2.f };
	std::vector<float> c = { 1.f, 3.f, 2.f, 6.f, 5.f, 4.f, 8.f, 10.f, 9.f, 1.f, 2.f };
	std::vector<float> d(size);

	size_t size3 = size * size * size;
	vex::Context ctx(vex::Filter::GPU);
	//x(0)(1)(1) = 1;
	vex::multivector<double, 3> X(ctx, size), Y(ctx, size);
	vex::copy(a, X(0));
	vex::copy(b, X(1));
	X(0) = 0;
	X(0)[0] = 1;

	X(1) = 1;
	X(1)[0] = 2;

	X(2) = 2;
	X(2)[0] = 3;


	vex::copy(X(0), a);
	vex::copy(X(1), b);
	vex::copy(X(2), c);

	for (size_t i = 0; i < a.size(); i++)
	{
		std::cout << "a[" << i << "] = " << a[i] << '\n';
	}

	for (size_t i = 0; i < b.size(); i++)
	{
		std::cout << "b[" << i << "] = " << b[i] << '\n';
	}
	for (size_t i = 0; i < c.size(); i++)
	{
		std::cout << "c[" << i << "] = " << c[i] << '\n';
	}


}



namespace L2S
{
	// (ii)(i, j, k)
	using RealType = float;
	typedef Eigen::Tensor<SpatialBlockField<RealType>, 3, Eigen::AutoAlign> SpatialField;
}

int main(void)
{
/*	
	vex::Context ctx(vex::Filter::GPU);


	if (!ctx) throw std::runtime_error("No devices available.");
	
	// Print out list of selected devices:
	std::cout << ctx << std::endl;
	//bench();
	//test();

	//test_multi_vector();
	*/
	//test();
	printf("EIGEN 1D \n");
	vex::Context a(vex::Filter::GPU);
	SpatialBlockField1D m(2);

	for (auto i = 0; i < m.size(); i++) {
#ifdef LSWM_WITH_VEXCL
		m[i] = i + 1;
		std::cout << "m(" << i << ") = " << m[i] << '\n';
#else
		m(i) = i + 1;
		std::cout << "m(" << i << ") = " << m(i) << '\n';
#endif	
	}

	using namespace L2S;

	using RealType = float;
	SpatialBlockField<RealType> test(1, 1, 1);
	test(0, 0, 0) = 1.f;
	auto & t = test(0, 0, 0);
	t = 5;
	SpatialField rho_;
	auto & _rho = rho_(0, 0, 0);
	_rho(0, 0, 0) = 1.f;
	//std::cout << "test(0,0,0) = " << test(0, 0, 0) << '\n';
	/*SpatialBlockField1D vexclT = test(0, 0);
	std::vector<float> cpuvex(4);
	vex::copy(vexclT, cpuvex);
	std::cout << "test(0,0,0) = " << cpuvex[0]<< '\n';*/


	/*
		SpatialField t(1);
		t(0).resize(1, 1, 1);
		t(0)(0, 0, 0) = 15.f;
		std::cout << "t.size = " << t.size() << '\n';
		std::cout << "t(" << 0 << ") = " << t(0)(0,0,0) << '\n';
		*/
	return 0;

}