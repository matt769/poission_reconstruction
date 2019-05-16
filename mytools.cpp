#include <igl/readOFF.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h> // errors if remove?
#include <igl/readPLY.h>
#include <igl/file_exists.h>
#include <igl/octree.h>
#include "nanoflann.hpp"
//#include <Eigen/Dense> // why not directly required? already included from igl above?
//#include <Eigen/Sparse>

#include "mytools.h"

void get_example_mesh(std::string const meshname, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& VN)
{


	std::vector<const char*> cands{
		"../../data/",
		"../../../data/",
		"../../../../data/",
		"../../../../../data/" };

	bool found = false;
	for (const auto& val : cands)
	{
		if (igl::file_exists(val + meshname))
		{
			std::cout << "loading example mesh from:" << val + meshname << "\n";

			if (igl::readOBJ(val + meshname, V, F)) {
				igl::per_vertex_normals(V, F, VN);
				found = 1;
				break;
			}
			else {
				std::cout << "file loading failed " << cands[0] + meshname << "\n";
			}
		}
	}

	if (!found) {
		std::cout << "cannot locate " << cands[0] + meshname << "\n";
		exit(1);
	}

}

// Is there really no way to just get point data only?
void get_point_data(std::string const meshname, Eigen::MatrixXd& V)
{
	Eigen::MatrixXi F;

	std::vector<const char*> cands{
		"../../data/",
		"../../../data/",
		"../../../../data/",
		"../../../../../data/" };

	bool found = false;
	for (const auto& val : cands)
	{
		if (igl::file_exists(val + meshname))
		{
			std::cout << "loading example mesh from:" << val + meshname << "\n";

			if (igl::readOBJ(val + meshname, V, F)) {
				found = 1;
				break;
			}
			else {
				std::cout << "file loading failed " << cands[0] + meshname << "\n";
			}
		}
	}

	if (!found) {
		std::cout << "cannot locate " << cands[0] + meshname << "\n";
		exit(1);
	}

}

void get_point_data(std::string const meshname, Eigen::MatrixXd& V, Eigen::MatrixXd& N)
{
	// not using these but necessary for the igl function readOBJ that includes the 'vn' type record
	Eigen::MatrixXd TC;
	Eigen::MatrixXi F;
	Eigen::MatrixXi FTC;
	Eigen::MatrixXi FN;

	std::vector<const char*> cands{
		"../../data/",
		"../../../data/",
		"../../../../data/",
		"../../../../../data/" };

	bool found = false;
	for (const auto& val : cands)
	{
		if (igl::file_exists(val + meshname))
		{
			std::cout << "loading example mesh from:" << val + meshname << "\n";

			if (igl::readOBJ(val + meshname, V, TC, N, F, FTC, FN)) {
				std::cout << "V = " << V.rows() << "x" << V.cols() << std::endl;
				std::cout << "N = " << N.rows() << "x" << N.cols() << std::endl;
				found = true;
				break;
			}
			else {
				std::cout << "file loading failed " << cands[0] + meshname << "\n";
			}
		}
	}

	if (!found) {
		std::cout << "cannot locate " << cands[0] + meshname << "\n";
		exit(1);
	}

}


// Get vertex id from resolution and indices
bool vertex_ij2idx(const Resolution& res, const int xIdx, const int yIdx, size_t &idx)
{
	if (yIdx >= res.y || xIdx >= res.x || yIdx < 0 || xIdx < 0)
		return false;

	idx = yIdx * res.x + xIdx;
	return true;
}



// Get vertex id from resolution and indices - 3D version
bool vertex_ijk2idx(const Resolution& res, const int xIdx, const int yIdx, const int zIdx, size_t& idx)
{
	if (yIdx >= res.y || xIdx >= res.x || zIdx >= res.z || yIdx < 0 || xIdx < 0 || zIdx < 0)
		return false;

	idx = zIdx * res.x * res.y + yIdx * res.x + xIdx;
	return true;
}

bool vertex_ijk2idx(const Resolution& res, const Resolution& xyz, size_t& idx)
{
	return vertex_ijk2idx(res, xyz.x, xyz.y, xyz.z, idx);
}


bool vertex_idx2ijk(const Resolution& res, const size_t idx, Resolution& xyz)
{
	if (idx > res.x*res.y*res.z)
		return false;


	xyz.x = idx % res.x;
	xyz.y = (idx / res.x) % res.y;
	xyz.z = idx / res.x / res.y;
	return true;
}






// Get indices of vertices of containing square
// x and y are position of any point in space
void xy2sq(const Eigen::MatrixXd &GV, const Resolution& res, const double x, const double y, Eigen::Matrix<size_t, 2, 2> &square)
{
	// TODO describe order of indices in 'square'

	const double cell_sz = (GV.col(0).maxCoeff() - GV.col(0).minCoeff()) / (res.x - 1);
	int xIdx = (int)((x - GV.col(0).minCoeff()) / cell_sz);
	int yIdx = (int)((y - GV.col(1).minCoeff()) / cell_sz);

	vertex_ij2idx(res, xIdx, yIdx, square(0, 0));
	vertex_ij2idx(res, xIdx + 1, yIdx, square(0, 1));
	vertex_ij2idx(res, xIdx, yIdx + 1, square(1, 0));
	vertex_ij2idx(res, xIdx + 1, yIdx + 1, square(1, 1));
}



void xyz2cube(const Eigen::MatrixXd& GV, const Resolution& res, const double x, const double y, const double z, std::vector<Eigen::Matrix<size_t, 2, 2>>& cube)
{

	cube.clear();

	const double cell_sz = (GV.col(0).maxCoeff() - GV.col(0).minCoeff()) / (res.x - 1);
	int xIdx = (int)((x - GV.col(0).minCoeff()) / cell_sz);
	int yIdx = (int)((y - GV.col(1).minCoeff()) / cell_sz);
	int zIdx = (int)((z - GV.col(2).minCoeff()) / cell_sz);

	Eigen::Matrix<size_t, 2, 2> square;

	vertex_ijk2idx(res, xIdx, yIdx, zIdx, square(0, 0));
	vertex_ijk2idx(res, xIdx + 1, yIdx, zIdx, square(0, 1));
	vertex_ijk2idx(res, xIdx, yIdx + 1, zIdx, square(1, 0));
	vertex_ijk2idx(res, xIdx + 1, yIdx + 1, zIdx, square(1, 1));
	cube.push_back(square);

	vertex_ijk2idx(res, xIdx, yIdx, zIdx + 1, square(0, 0));
	vertex_ijk2idx(res, xIdx + 1, yIdx, zIdx + 1, square(0, 1));
	vertex_ijk2idx(res, xIdx, yIdx + 1, zIdx + 1, square(1, 0));
	vertex_ijk2idx(res, xIdx + 1, yIdx + 1, zIdx + 1, square(1, 1));
	cube.push_back(square);


}





void get_grid(const Eigen::MatrixXd& V, const int depth, Eigen::MatrixXd& GV, Eigen::MatrixXi& GE, Resolution& res)
{
	const size_t extra_layers = 3;	// applied on each side

	// find bounding area
	Eigen::RowVector3d BBmin = V.colwise().minCoeff();
	Eigen::RowVector3d BBmax = V.colwise().maxCoeff();
	Eigen::RowVector3d BBrange = BBmax - BBmin;

	/*const Eigen::RowVector3d adjBB = BBrange;
	BBmin -= adjBB;
	BBmax += adjBB;
	BBrange = BBmax - BBmin;*/

	// create grid (square mesh) across bounding area
	// think of resolution as the result of multiple equal axis splits
	// or desired number of vertices along longest side
	const int splits = pow(2, depth);

	// not all sides of the bounding area may be equal
	// split the largest by the split number-1 i.e. n vertices, n-1 steps between them all
	const double step = BBrange.maxCoeff() / (double)(splits-1);
	// how many *whole* steps cover each side
	// TODO clean up following few lines
	Eigen::RowVector3i tmpRes = (BBrange / step).cast<int>();
	tmpRes += 2 * Eigen::RowVector3i::Ones(3);
	// Add extra layers (each side)
	tmpRes += 2 * extra_layers * Eigen::RowVector3i::Ones(3);
	// Adjust min (starting point of grid) to account for the added layers
	BBmin -= Eigen::RowVector3d::Constant(step * (double)extra_layers);

	res.x = tmpRes(0);
	res.y = tmpRes(1);
	res.z = tmpRes(2);

	// now create grid vertices
	GV.resize(res.x * res.y * res.z, 3);

	int vIdx = 0;
	for (int zIdx = 0; zIdx < res.z; zIdx++)
	{
		double z = BBmin(2) + (step * zIdx);

		for (int yIdx = 0; yIdx < res.y; yIdx++)
		{
			double y = BBmin(1) + (step * yIdx);

			for (int xIdx = 0; xIdx < res.x; xIdx++)
			{
				double x = BBmin(0) + (step * xIdx);
				GV.row(vIdx) = Eigen::RowVector3d(x, y, z);
				vIdx++;
			}
		}
	}

	// adjust grid so that center matches object center?



}

// should this take a full list index or x,y,x indices?
//void get_grid_neighbours(const Eigen::MatrixXd& GV, const Resolution& res, const size_t idx, Eigen::VectorXi neighbourIdx)
//{
//
//}

// separate functions for returning neighbours in each dimension?




void construct_laplacian(const Resolution& res, Eigen::MatrixXd& GV, Eigen::SparseMatrix<double>& L)
{
	// ONLY SUPPORTS 2D GRID
	int m = GV.rows();
	L.resize(m, m);
	// because it's a regular grid, we don't need to look for neighbours
	// note that the grid is built by iterating through z, then y, then x
	// for a 2D grid vith a list of vertices V, the indices (in grid form) are:
	//	2	5	8	11
	//	1	4	7	10
	//	0	3	6	9
	// so the neighbours of n are: n-1, n+1 (on same column), n-y, n+y (on same row)

	// reserve space for 5 non-zero elements on each row
	L.reserve(Eigen::VectorXi::Constant(m, 9));

	for (int idx = 0; idx < m; idx++)
	{
		// note that idx is the vertex index, and the row index of the laplacian
		// and there are res.x-1 squares on the x side
		Resolution xyz;
		if (!vertex_idx2ijk(res, idx, xyz))
		{
			std::cout << "Invalid index\n";
			return;
		}

		const Resolution offsets[6] = { {0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0} };


		int neighbourCount = 0;
		for (int offsetIdx = 0; offsetIdx < 6; offsetIdx++)
		{
			size_t nIdx;
			if (!vertex_ijk2idx(res, xyz + offsets[offsetIdx], nIdx))
			{
				continue;
			}

			neighbourCount++;

		}

		double coeff = 1.0 / (double)neighbourCount;
		L.insert(idx, idx) = -1.0;

		for (int offsetIdx = 0; offsetIdx < 6; offsetIdx++)
		{
			size_t nIdx;
			if (!vertex_ijk2idx(res, xyz + offsets[offsetIdx], nIdx))
			{
				continue;
			}

			L.insert(idx, nIdx) = coeff;

		}
	}


	L.makeCompressed();

}


void construct_divergence(const Resolution& res,
							Eigen::SparseMatrix<double, Eigen::RowMajor>& Dx,
							Eigen::SparseMatrix<double, Eigen::RowMajor>& Dy,
							Eigen::SparseMatrix<double, Eigen::RowMajor>& Dz)
{
	// ONLY SUPPORTS 2D GRID
	int m = res.x * res.y * res.z;
	Dx.resize(m, m);	// is 'resize' actually required here?
	Dy.resize(m, m);
	Dz.resize(m, m);
	// because it's a regular grid, we don't need to look for neighbours
	// note that the grid is built by iterating through z, then y, then x
	// for a 2D grid vith a list of vertices V, the indices (in grid form) are:
	//	2	5	8	11
	//	1	4	7	10
	//	0	3	6	9
	// so the neighbours of n are: n-1, n+1 (on same column), n-y, n+y (on same row)

	// reserve space for 2 non-zero elements on each row
	Dx.reserve(2 * m);
	Dy.reserve(2 * m);
	Dz.reserve(2 * m);

	for (int idx = 0; idx < m; idx++)
	{

		Resolution xyz;
		if (!vertex_idx2ijk(res, idx, xyz))
		{
			std::cout << "Invalid index\n";
			return;
		}

		const Resolution offsets[3] = { {0, 0, 1}, {0, 1, 0}, {1, 0, 0} };

		Dx.insert(idx, idx) = -1.0;
		Dy.insert(idx, idx) = -1.0;
		Dz.insert(idx, idx) = -1.0;

		for (int offsetIdx = 0; offsetIdx < 3; offsetIdx++)
		{
			size_t nIdx;
			if (!vertex_ijk2idx(res, xyz + offsets[offsetIdx], nIdx))
			{
				continue;
			}

			switch (offsetIdx)
			{
			case 0:
				Dz.insert(idx, nIdx) = 1.0;
				break;
			case 1:
				Dy.insert(idx, nIdx) = 1.0;
				break;
			case 2:
				Dx.insert(idx, nIdx) = 1.0;
				break;
			default:
				break;
			}

		}

		// -3 for all, assumes that value is zero on 'virtual' neighbouring points off the grid
		// one -3 is in Dx, one in Dy, one in Dz
	}

	Dx.makeCompressed();
	Dy.makeCompressed();
	Dz.makeCompressed();
}

void compute_grid_normals(
	const Eigen::MatrixXd &V, 
	const Eigen::MatrixXd &N, 
	const Eigen::MatrixXd &GV, 
	const size_t k, 
	Eigen::MatrixXd &weightedNormals)
{
	weightedNormals = Eigen::MatrixXd::Zero(GV.rows(), 3);

	Eigen::VectorXd totalWeight = Eigen::VectorXd::Zero(GV.rows());
	double gridSize = (GV.row(0) - GV.row(1)).norm();
	double gridDiagSquared = 2 * gridSize * gridSize;

	typedef nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixXd >  kd_tree_t;
	kd_tree_t mat_index(GV, 10 /* max leaf */);
	mat_index.index->buildIndex();

	for (int searchPointIdx = 0; searchPointIdx < V.rows(); searchPointIdx++) {
		//std::cout << searchPointIdx << "\n";
		Eigen::RowVector3d searchPoint = V.row(searchPointIdx);

		// create a query object
		std::vector<size_t> neighbourIdx(k);
		std::vector<double> distsSqr(k);

		nanoflann::KNNResultSet<double> knnSearchResult(k);
		knnSearchResult.init(neighbourIdx.data(), distsSqr.data());

		// find neighbours
		mat_index.index->findNeighbors(knnSearchResult, searchPoint.data(), nanoflann::SearchParams(50));

		Eigen::VectorXd weights(k);
		for (size_t i = 0; i < neighbourIdx.size(); i++)
		{
			//double weight = 1.0 / distsSqr[i];
			//std::cout << N.row(searchPointIdx) << "\n";
			weights(i) = 1.0 - distsSqr[i] / gridDiagSquared;
			//V_nn.row(i) = V.row(neighbourIdx[i]);
		}
		weights /= weights.sum();
		for (size_t i = 0; i < neighbourIdx.size(); i++)
		{
			weightedNormals.row(neighbourIdx[i]) += weights(i) * N.row(searchPointIdx);
			totalWeight(neighbourIdx[i]) += weights(i);
		}

	}

	double maxNormalSize = weightedNormals.rowwise().norm().maxCoeff();
	weightedNormals = weightedNormals * gridSize / maxNormalSize;
}



// Only works if M and K are 2D
void conv2d(const Eigen::MatrixXd &M, const Eigen::MatrixXd &K, const Resolution& res, Eigen::MatrixXd &MK)
{
	MK = Eigen::MatrixXd::Zero(M.rows(), M.cols());
	//std::cout << "In conv2d...\n";
	for (size_t yIdx = 0; yIdx < res.y; yIdx++)
		for (size_t xIdx = 0; xIdx < res.x; xIdx++)
		{
			size_t o;
			if (!vertex_ij2idx(res, xIdx, yIdx, o))
			{
				std::cout << "ERROR: Invalid vertex indices!!!\n";
				return;
			}
			for (size_t ik = 0; ik < K.rows(); ik++)
				for (size_t jk = 0; jk < K.cols(); jk++)
				{
					int xIdx_NB = xIdx + ik - K.rows() / 2;
					int yIdx_NB = yIdx + jk - K.cols() / 2;
					size_t neigh;
					bool valid_neigh = vertex_ij2idx(res, xIdx_NB, yIdx_NB, neigh);
					if (valid_neigh)
						MK.row(o) += K(ik, jk) * M.row(neigh);
				}
		}
}


// M is a list of the normal vectors that will be smoothed
// K is the pre-calculated smoothing kernel in 1D - it will be applied to each dimension
// CHECK why results seem to be large - should be normalised (as long as K is)
void conv3d(const Eigen::MatrixXd& M, const Eigen::VectorXd& K, const Resolution& res, Eigen::MatrixXd& MK)
{
	MK = Eigen::MatrixXd(M);
	Eigen::MatrixXd MK_tmp = Eigen::MatrixXd::Zero(M.rows(), M.cols());

	// Apply in x dimension
	for (size_t xIdx = 0; xIdx < res.x; xIdx++)
	{
		for (size_t yIdx = 0; yIdx < res.y; yIdx++)
		{
			for (size_t zIdx = 0; zIdx < res.z; zIdx++)
			{
				size_t o;
				if (!vertex_ijk2idx(res, xIdx, yIdx, zIdx, o))
				{
					std::cout << "ERROR: Invalid vertex indices!!!\n";
					return;
				}
				for (size_t ik = 0; ik < K.rows(); ik++)
				{
					int xIdx_NB = xIdx + ik - K.rows() / 2;
					size_t neigh;
					bool valid_neigh = vertex_ijk2idx(res, xIdx_NB, yIdx, zIdx, neigh);
					if (valid_neigh)
						MK_tmp.row(o) += K(ik) * MK.row(neigh);
				}
			}
		}
	}
	MK = MK_tmp;

	// Apply in y dimension
	for (size_t xIdx = 0; xIdx < res.x; xIdx++)
	{
		for (size_t yIdx = 0; yIdx < res.y; yIdx++)
		{
			for (size_t zIdx = 0; zIdx < res.z; zIdx++)
			{
				size_t o;
				if (!vertex_ijk2idx(res, xIdx, yIdx, zIdx, o))
				{
					std::cout << "ERROR: Invalid vertex indices!!!\n";
					return;
				}
				for (size_t ik = 0; ik < K.rows(); ik++)
				{
					int yIdx_NB = yIdx + ik - K.rows() / 2;
					size_t neigh;
					bool valid_neigh = vertex_ijk2idx(res, xIdx, yIdx_NB, zIdx, neigh);
					if (valid_neigh)
						MK_tmp.row(o) += K(ik) * MK.row(neigh);
				}
			}
		}
	}
	MK = MK_tmp;

	// Apply in z dimension
	for (size_t xIdx = 0; xIdx < res.x; xIdx++)
	{
		for (size_t yIdx = 0; yIdx < res.y; yIdx++)
		{
			for (size_t zIdx = 0; zIdx < res.z; zIdx++)
			{
				size_t o;
				if (!vertex_ijk2idx(res, xIdx, yIdx, zIdx, o))
				{
					std::cout << "ERROR: Invalid vertex indices!!!\n";
					return;
				}
				for (size_t ik = 0; ik < K.rows(); ik++)
				{
					int zIdx_NB = zIdx + ik - K.rows() / 2;
					size_t neigh;
					bool valid_neigh = vertex_ijk2idx(res, xIdx, yIdx, zIdx_NB, neigh);
					if (valid_neigh)
						MK_tmp.row(o) += K(ik) * MK.row(neigh);
				}
			}
		}
	}
	MK = MK_tmp;
}


double lin_interp(const double lx, const double lv, const double rx, const double rv, const double x)
{
	return ((rx - x) * lv + (x - lx) * rv) / (rx - lx);
}


double compute_isovalue(
	const Eigen::MatrixXd &V,
	const Eigen::MatrixXd &GV,
	const Resolution& res,
	const Eigen::VectorXd &Chi)
{
	double isoval = 0;
	std::vector<Eigen::Matrix<size_t, 2, 2>> cube;
	for (size_t i = 0; i < V.rows(); i++)
	{
		xyz2cube(GV, res, V(i, 0), V(i, 1), V(i, 2), cube);
		std::vector<double> chi_xy;
		for (Eigen::Matrix<size_t, 2, 2> sq : cube)
		{
			double x_bottom_chi = lin_interp(GV(sq(0, 0), 0), Chi(sq(0, 0)), GV(sq(0, 1), 0), Chi(sq(0, 1)), V(i, 0));
			double x_top_chi = lin_interp(GV(sq(1, 0), 0), Chi(sq(1, 0)), GV(sq(1, 1), 0), Chi(sq(1, 1)), V(i, 0));
			chi_xy.push_back(lin_interp(GV(sq(0, 0), 1), x_bottom_chi, GV(sq(1, 0), 1), x_top_chi, V(i, 1)));
		}

		double chi = lin_interp(GV(cube[0](0, 0), 2), chi_xy[0], GV(cube[1](0, 0), 2), chi_xy[1], V(i, 2));


		isoval += chi;
	}
	isoval /= V.rows();
	return isoval;
}