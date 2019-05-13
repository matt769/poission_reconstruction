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
bool vertex_ij2idx(const Eigen::RowVector3i &res, const int r, const int c, size_t &idx)
{
	if (r >= res(0) || c >= res(1) || r < 0 || c < 0)
		return false;

	idx = c * res(1) + r;
	return true;
}


// Get vertex id from resolution and indices
void xy2sq(const Eigen::MatrixXd &GV, const Eigen::RowVector3i &res, const double x, const double y, Eigen::Matrix<size_t, 2, 2> &square)
{
	const double cell_sz = (GV.col(0).maxCoeff() - GV.col(0).minCoeff()) / (res(1) - 1);
	double c = (x - GV.col(0).minCoeff()) / cell_sz;
	double r = (y - GV.col(1).minCoeff()) / cell_sz;


	vertex_ij2idx(res, int(r), int(c), square(0, 0));
	vertex_ij2idx(res, int(r) + 1, int(c), square(0, 1));
	vertex_ij2idx(res, int(r), int(c) + 1, square(1, 0));
	vertex_ij2idx(res, int(r) + 1, int(c) + 1, square(1, 1));
}


void get_grid(const Eigen::MatrixXd& V, const int depth, Eigen::MatrixXd& GV, Eigen::MatrixXi& GE, Eigen::RowVector3i& res)
{
	const size_t extra_layers = 20;

	// find bounding area
	Eigen::RowVector3d BBmin = V.colwise().minCoeff();
	Eigen::RowVector3d BBmax = V.colwise().maxCoeff();
	Eigen::RowVector3d BBrange = BBmax - BBmin;

	/*const Eigen::RowVector3d adjBB = BBrange;
	BBmin -= adjBB;
	BBmax += adjBB;
	BBrange = BBmax - BBmin;*/

	// create grid (square mesh) across bounding area
	// remember that there will be points on the edges
	// what do we want as the resolution?
	//int depth = 5; // if we think of resolution as the result of multiple equal axis splits
	const int splits = pow(2, depth);

	// not all sides of the bounding area may be equal
	// split the largest by the resolution
	const double step = BBrange.maxCoeff() / (double)splits;
	// how many whole steps cover each side
	res = (BBrange / step).cast<int>() + extra_layers * Eigen::RowVector3i::Ones(3);
	res(2) = 1;
	BBmin.block(0, 0, 1, 2) -= Eigen::RowVector2d::Constant(step * extra_layers / 2);


	// now create grid vertices
	//Eigen::MatrixXd GV(res(0) * res(1) * res(2), 3);
	GV.resize(res(0) * res(1) * res(2), 3);

	int rowIdx = 0;
	for (int xIdx = 0; xIdx < res(0); xIdx++)
	{
		double x = BBmin(0) + (step * xIdx);

		for (int yIdx = 0; yIdx < res(1); yIdx++)
		{
			double y = BBmin(1) + (step * yIdx);

			for (int zIdx = 0; zIdx < res(2); zIdx++)
			{
				double z = BBmin(2) + (step * zIdx);
				GV.row(rowIdx) = Eigen::RowVector3d(x, y, z);
				rowIdx++;
			}
		}
	}

}

void construct_laplacian(Eigen::MatrixXd& GV, Eigen::SparseMatrix<double>& L, Eigen::RowVector3i& res)
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
	L.reserve(Eigen::VectorXi::Constant(m, 5));

	for (int idx = 0; idx < m; idx++)
	{
		// note that idx is the vertex index, and the row index of the laplacian
		bool onLowerEdgeX = false;
		bool onUpperEdgeX = false;
		bool onLowerEdgeY = false;
		bool onUpperEdgeY = false;
		int neighbourCount = 4;

		if (idx < res(1))
		{
			onLowerEdgeX = true;
			neighbourCount -= 1;
		}
		else if (idx >= m - res(1))
		{
			onUpperEdgeX = true;
			neighbourCount -= 1;
		}

		if (idx % res(1) == 0)
		{
			onLowerEdgeY = true;
			neighbourCount -= 1;
		}
		else if ((idx + 1) % res(1) == 0)
		{
			onUpperEdgeY = true;
			neighbourCount -= 1;
		}
		
		double coeff = 1.0 / (double)neighbourCount;
	

		// add diag
		L.insert(idx, idx) = -1;
		// add neighbours
		if (!onLowerEdgeX)
		{
			L.insert(idx, idx - res(1)) = coeff;
		}

		if (!onUpperEdgeX)
		{
			L.insert(idx, idx + res(1)) = coeff;
		}

		if (!onLowerEdgeY)
		{
			L.insert(idx, idx - 1) = coeff;
		}

		if (!onUpperEdgeY)
		{
			L.insert(idx, idx + 1) = coeff;
		}


	}

	L.makeCompressed();

	//std::cout << L.block(0, 0, 5, 5) << std::endl;
	//std::cout << L << std::endl;

	// probably more efficient way to do this
	// see also, construction from triplets https://eigen.tuxfamily.org/dox/group__TutorialSparse.html

}


void construct_divergence(Eigen::RowVector3i& res, 
							Eigen::SparseMatrix<double, Eigen::RowMajor>& Dx,
							Eigen::SparseMatrix<double, Eigen::RowMajor>& Dy)
{
	// ONLY SUPPORTS 2D GRID
	int m = res(0)*res(1);
	Dx.resize(m, m);
	Dy.resize(m, m);
	// because it's a regular grid, we don't need to look for neighbours
	// note that the grid is built by iterating through z, then y, then x
	// for a 2D grid vith a list of vertices V, the indices (in grid form) are:
	//	2	5	8	11
	//	1	4	7	10
	//	0	3	6	9
	// so the neighbours of n are: n-1, n+1 (on same column), n-y, n+y (on same row)

	// reserve space for 5 non-zero elements on each row
	Dx.reserve(2 * m);
	Dy.reserve(2 * m);

	for (int idx = 0; idx < m; idx++)
	{
		bool onUpperEdgeX = false;
		bool onUpperEdgeY = false;
		int neighbourCount = 2;

		if (idx >= m - res(1))
		{
			onUpperEdgeX = true;
		}

		if ((idx + 1) % res(1) == 0)
		{
			onUpperEdgeY = true;
		}

		// add diag
		// -2 for all, assumes that value is zero on 'virtual' neighbouring points off the grid
		Dx.insert(idx, idx) = -1.0;
		Dy.insert(idx, idx) = -1.0;
		// add neighbours
		if (!onUpperEdgeX)
		{
			Dx.insert(idx, idx + res(1)) = 1.0;
		}
		if (!onUpperEdgeY)
		{
			Dy.insert(idx, idx + 1) = 1.0;
		}

	}

	Dx.makeCompressed();
	Dy.makeCompressed();

}

void compute_grid_normals(const Eigen::MatrixXd &V, const Eigen::MatrixXd &N, const Eigen::MatrixXd &GV, const size_t k, Eigen::MatrixXd &weightedNormals)
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

		// set K nearest samples
		//const size_t k = 4;		// TODO CHANGE THIS TO DO RADIUS SEARCH

		// create a query object
		std::vector<size_t> neighbourIdx(k);
		std::vector<double> distsSqr(k);

		nanoflann::KNNResultSet<double> knnSearchResult(k);
		knnSearchResult.init(neighbourIdx.data(), distsSqr.data());

		// find neighbours
		mat_index.index->findNeighbors(knnSearchResult, searchPoint.data(), nanoflann::SearchParams(50));
		//Eigen::MatrixXd V_nn(neighbourIdx.size(), 3);
		for (size_t i = 0; i < neighbourIdx.size(); i++)
		{
			//double weight = 1.0 / distsSqr[i];
			//std::cout << N.row(searchPointIdx) << "\n";
			double weight = 1.0 - distsSqr[i] / gridDiagSquared;
			weightedNormals.row(neighbourIdx[i]) += weight * N.row(searchPointIdx);
			totalWeight(neighbourIdx[i]) += weight;
			//V_nn.row(i) = V.row(neighbourIdx[i]);
		}

	}

	double maxNormalSize = weightedNormals.rowwise().norm().maxCoeff();
	weightedNormals = weightedNormals * gridSize / maxNormalSize;
}



// Only works if M and K are 2D
void conv2d(const Eigen::MatrixXd &M, const Eigen::MatrixXd &K, const Eigen::RowVector3i &res, Eigen::MatrixXd &MK)
{
	MK = Eigen::MatrixXd::Zero(M.rows(), M.cols());

	for (size_t i = 0; i < res(1); i++)
		for (size_t j = 0; j < res(0); j++)
		{
			size_t o;
			if (!vertex_ij2idx(res, i, j, o))
			{
				std::cout << "ERROR: Invalid vertex indices!!!\n";
				return;
			}
			for (size_t ik = 0; ik < K.rows(); ik++)
				for (size_t jk = 0; jk < K.cols(); jk++)
				{
					int r = i + ik - K.rows() / 2;
					int c = j + jk - K.cols() / 2;
					size_t neigh;
					bool valid_neigh = vertex_ij2idx(res, r, c, neigh);
					if (valid_neigh)
						MK.row(o) += K(ik, jk) * M.row(neigh);
				}
		}
}

double lin_interp(const double lx, const double lv, const double rx, const double rv, const double x)
{
	return ((rx - x) * lv + (x - lx) * rv) / (rx - lx);
}


double compute_isovalue(
	const Eigen::MatrixXd &V,
	const Eigen::MatrixXd &GV,
	const Eigen::RowVector3i &res,
	const Eigen::VectorXd &Chi)
{
	double isoval = 0;
	Eigen::Matrix<size_t, 2, 2> sq;
	for (size_t i = 0; i < V.rows(); i++)
	{
		xy2sq(GV, res, V(i, 0), V(i, 1), sq);

		double x_bottom_chi = lin_interp(GV(sq(0, 0), 1), Chi(sq(0, 0)), GV(sq(0, 1), 1), Chi(sq(0, 1)), V(i, 1));
		double x_top_chi = lin_interp(GV(sq(1, 0), 1), Chi(sq(1, 0)), GV(sq(1, 1), 1), Chi(sq(1, 1)), V(i, 1));
		double chi = lin_interp(GV(sq(0, 0), 0), x_bottom_chi, GV(sq(1, 0), 0), x_top_chi, V(i, 0));

		isoval += chi;
	}
	isoval /= V.rows();
	return isoval;
}