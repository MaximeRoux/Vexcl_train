#include <stdio.h>
#include <stdexcept>
#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


#define VEXCL_BACKEND_CUDA

#include <vexcl/vexcl.hpp>
#include <vexcl/vector_view.hpp>


using RealType = float;

typedef vex::vector<RealType> SpatialBlockField1D;


const vex::Context& ctx() {
	static vex::Context ctx(vex::Filter::GPU);
	return ctx;
}


namespace L2S
{

	template<typename RealType>
	class SpatialBlockField
	{

	public:

		SpatialBlockField(const int nx, const int ny, const int nz) :nx_(nx),
			ny_(ny),
			nz_(nz)
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
					data_(i, j) = SpatialBlockField1D(ctx(), nz_);

				}
			}
		}


		inline auto & operator()(const int i, const int j)
		{
			return data_(i, j);
		}

		inline const auto & operator()(const int i, const int j) const
		{
			return data_(i, j);
		}

		inline auto & operator()(const int i, const int j, const int k)
		{
			
			std::cout << " vector vexcl size(data(i,j)) = " << data_(i, j).size() << std::endl;
			auto & dij = data_(i, j);
			std::cout << " vector vexcl size(dij) = " << dij.size() << std::endl;
			std::cout << " vector vexcl size data = " << data_.size() << std::endl;
			
			return data_(i, j)[k];
		}

		inline const auto & operator()(const int i, const int j, const int k) const
		{
			return data_(i, j)[k];
		}


	private:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW // To force an object of this class to be allocated as aligned


		const int packetSize_ = Eigen::internal::packet_traits<RealType>::size;

		int nx_;
		int ny_;
		int nz_;

		/* Padding along x, y and z */
		int px_ = 0;
		int py_ = 0;
		int pz_ = 0;

		int n_;
		// (i,j)(k)
		Eigen::Array<SpatialBlockField1D, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> data_;

	};

}


namespace L2S
{
	// (ii)(i, j, k)
	using RealType = float;
	typedef Eigen::Tensor<SpatialBlockField<RealType>, 1, Eigen::AutoAlign> SpatialField;

}

int main(void)
{

	printf("EIGEN 1D \n");
	SpatialBlockField1D m(ctx(), 2);

	for (auto i = 0; i < m.size(); i++) {
		m[i] = i + 1;
		std::cout << "m(" << i << ") = " << m[i] << '\n';

	}

	using namespace L2S;

	using RealType = float;
	SpatialBlockField<RealType> test(1, 1, 1);
	test(0, 0, 0) = 1.f;
	SpatialBlockField1D vexclT = test(0, 0);
	std::vector<float> cpuvex(4);
	vex::copy(vexclT, cpuvex);
	std::cout << "test(0,0,0) = " << cpuvex[0] << '\n';

	return 0;

}