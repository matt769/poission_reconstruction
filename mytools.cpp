#include <igl/readOFF.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h> // errors if remove?
#include <igl/readPLY.h>
#include <igl/file_exists.h>
#include <igl/octree.h>
#include "nanoflann.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

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
bool vertex_ijk2idx(const Resolution& res, const int xIdx, const int yIdx, const int zIdx, size_t& idx)
{
	if (yIdx >= res.y || xIdx >= res.x || zIdx >= res.z || yIdx < 0 || xIdx < 0 || zIdx < 0)
		return false;

	idx = zIdx * res.x * res.y + yIdx * res.x + xIdx;
	return true;
}

// Takes Resolution object instead of separate x,y,z
bool vertex_ijk2idx(const Resolution& res, const Resolution& xyz, size_t& idx)
{
	return vertex_ijk2idx(res, xyz.x, xyz.y, xyz.z, idx);
}

// Get vertex x,y,z indices (as Resolution object) based on its index in the full list and the grid resolution
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
	// TODO could do with some explanation

	const double cell_sz = (GV.col(0).maxCoeff() - GV.col(0).minCoeff()) / (res.x - 1);
	int xIdx = (int)((x - GV.col(0).minCoeff()) / cell_sz);
	int yIdx = (int)((y - GV.col(1).minCoeff()) / cell_sz);

	vertex_ijk2idx(res, xIdx, yIdx, 0, square(0, 0));
	vertex_ijk2idx(res, xIdx + 1, yIdx, 0, square(0, 1));
	vertex_ijk2idx(res, xIdx, yIdx + 1, 0, square(1, 0));
	vertex_ijk2idx(res, xIdx + 1, yIdx + 1, 0, square(1, 1));
}

// like xy2sq but returns the 2 squares that make up the cube containing the searched for point
void xyz2cube(const Eigen::MatrixXd& GV, const Resolution& res, const double x, const double y, const double z, std::vector<Eigen::Matrix<size_t, 2, 2>>& cube)
{

	// TODO could do with some explanation
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



void get_grid(const Eigen::MatrixXd& V, const int depth, const size_t extra_layers, Eigen::MatrixXd& GV, Eigen::MatrixXi& GE, Resolution& res)
{

	// find bounding area
	Eigen::RowVector3d BBmin = V.colwise().minCoeff();
	Eigen::RowVector3d BBmax = V.colwise().maxCoeff();
	Eigen::RowVector3d BBrange = BBmax - BBmin;

	bool twoD = false;
	if (abs(BBrange(2)) < 0.00000001)
	{
		twoD = true;
	}

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
	tmpRes += 2 * Eigen::RowVector3i::Ones(3); // add 2 extra layers to ensure coverage
	// Add extra layers (each side)
	tmpRes += 2 * extra_layers * Eigen::RowVector3i::Ones(3);

	// Adjust min (starting point of grid) to account for the added layers
	BBmin -= Eigen::RowVector3d::Constant(step * (double)extra_layers);

	if (twoD)
	{
		tmpRes(2) = 1;
		BBmin(2) = 0.0;
	}

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


void construct_laplacian(const Resolution& res, Eigen::MatrixXd& GV, Eigen::SparseMatrix<double>& L)
{

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

		const Resolution offsets[6] = { {0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0} };

		//Dx.insert(idx, idx) = -1.0;
		//Dy.insert(idx, idx) = -1.0;
		//Dz.insert(idx, idx) = -1.0;

		for (int offsetIdx = 0; offsetIdx < 6; offsetIdx++)
		{
			size_t nIdx;
			if (!vertex_ijk2idx(res, xyz + offsets[offsetIdx], nIdx))
			{
				continue;
			}

			switch (offsetIdx)
			{
			case 0:
				Dz.insert(idx, nIdx) = -0.5;
				break;
			case 1:
				Dz.insert(idx, nIdx) = 0.5;
				break;
			case 2:
				Dy.insert(idx, nIdx) = -0.5;
				break;
			case 3:
				Dy.insert(idx, nIdx) = 0.5;
				break;
			case 4:
				Dx.insert(idx, nIdx) = -0.5;
				break;
			case 5:
				Dx.insert(idx, nIdx) = 0.5;
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
	Eigen::MatrixXd &GN)
{
	GN = Eigen::MatrixXd::Zero(GV.rows(), 3);

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
			double weight = 1.0 - distsSqr[i] / gridDiagSquared;
			weights(i) = std::max(0.0, weight); //
			//V_nn.row(i) = V.row(neighbourIdx[i]);
		}
		weights /= weights.sum();
		for (size_t i = 0; i < neighbourIdx.size(); i++)
		{
			GN.row(neighbourIdx[i]) += weights(i) * N.row(searchPointIdx);
			totalWeight(neighbourIdx[i]) += weights(i);
		}

	}

	double maxNormalSize = GN.rowwise().norm().maxCoeff();
	GN = GN * gridSize / maxNormalSize;
}

// this version will spread the normals to the grid points containing the source vertex
// will spread to 4 (square) if 2D and 8 (cube) if 3D
void compute_grid_normals(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXd& N,
	const Eigen::MatrixXd& GV,
	const Resolution& res,
	Eigen::MatrixXd& GN)
{
	GN = Eigen::MatrixXd::Zero(GV.rows(), 3);
	double gridSize = (GV.row(0) - GV.row(1)).norm();
	

	if (res.z != 1)
	{
		// 3D
		double gridDiag = sqrt(3) * gridSize;
		for (int i = 0; i < V.rows(); i++)
		{
			// get the grid points that this normal should be spread to
			std::vector<Eigen::Matrix<size_t, 2, 2>> cube;
			xyz2cube(GV, res, V(i, 0), V(i, 1), V(i, 2), cube);
			// for each grid point, take an equal share (for now) of the normal
			//double weight = 1.0 / 8.0;
			double totalWeight = 0.0;

			for (auto square : cube)
			{
				// doesn't seem to be a reshape option for the matrix, old version?
				Eigen::RowVector3d diff;
				double dist;
				double weight;
				
				diff = V.row(i) - GV.row(square(0, 0));
				dist = diff.norm();
				weight = 1.0 - dist / gridDiag;
				GN.row(square(0, 0)) += N.row(i) * weight;
				totalWeight += weight;

				diff = V.row(i) - GV.row(square(0, 1));
				dist = diff.norm();
				weight = 1.0 - dist / gridDiag;
				GN.row(square(0, 1)) += N.row(i) * weight;
				totalWeight += weight;

				diff = V.row(i) - GV.row(square(1, 0));
				dist = diff.norm();
				weight = 1.0 - dist / gridDiag;
				GN.row(square(1, 0)) += N.row(i) * weight;
				totalWeight += weight;

				diff = V.row(i) - GV.row(square(1, 1));
				dist = diff.norm();
				weight = 1.0 - dist / gridDiag;
				GN.row(square(1, 1)) += N.row(i) * weight;
				totalWeight += weight;
			}

			for (auto square : cube)
			{
				GN.row(square(0, 0)) /= totalWeight;
				GN.row(square(0, 1)) /= totalWeight;
				GN.row(square(1, 0)) /= totalWeight;
				GN.row(square(1, 1)) /= totalWeight;
			}

		}
	}
	else
	{
		// 2D
		double gridDiag = sqrt(2) * gridSize;
		for (int i = 0; i < V.rows(); i++)
		{
			// get the grid points that this normal should be spread to
			Eigen::Matrix<size_t, 2, 2> square;
			xy2sq(GV, res, V(i, 0), V(i, 1), square);
			// for each grid point, take an equal share (for now) of the normal
			Eigen::RowVector3d diff;
			double dist;
			double weight;
			double totalWeight = 0.0;

			diff = V.row(i) - GV.row(square(0, 0));
			dist = diff.norm();
			weight = 1.0 - dist / gridDiag;
			GN.row(square(0, 0)) += N.row(i) * weight;
			totalWeight += weight;

			diff = V.row(i) - GV.row(square(0, 1));
			dist = diff.norm();
			weight = 1.0 - dist / gridDiag;
			GN.row(square(0, 1)) += N.row(i) * weight;
			totalWeight += weight;

			diff = V.row(i) - GV.row(square(1, 0));
			dist = diff.norm();
			weight = 1.0 - dist / gridDiag;
			GN.row(square(1, 0)) += N.row(i) * weight;
			totalWeight += weight;

			diff = V.row(i) - GV.row(square(1, 1));
			dist = diff.norm();
			weight = 1.0 - dist / gridDiag;
			GN.row(square(1, 1)) += N.row(i) * weight;
			totalWeight += weight;

			GN.row(square(0, 0)) /= totalWeight;
			GN.row(square(0, 1)) /= totalWeight;
			GN.row(square(1, 0)) /= totalWeight;
			GN.row(square(1, 1)) /= totalWeight;
		}
	}


	double maxNormalSize = GN.rowwise().norm().maxCoeff();
	GN = GN * gridSize / maxNormalSize;

}


// M is a list of the normal vectors that will be smoothed
// K is the pre-calculated smoothing kernel in 1D - it will be applied to each dimension
// CHECK why results seem to be large - should be normalised (as long as K is)
void apply_convolution(const Eigen::MatrixXd& M, const Eigen::VectorXd& K, const Resolution& res, Eigen::MatrixXd& MK)
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
	MK_tmp = Eigen::MatrixXd::Zero(M.rows(), M.cols());

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
	MK_tmp = Eigen::MatrixXd::Zero(M.rows(), M.cols());

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


bool solve_poisson_equation(const Eigen::SparseMatrix<double> L , const Eigen::MatrixXd DGV, Eigen::VectorXd& x)
{
	bool success = true;
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;

	solver.compute(L);
	if (solver.info() != Eigen::Success) {
		success = false;
		std::cout << "Decomposition failed\n";
	}
	x = -solver.solve(DGV);
	if (solver.info() != Eigen::Success) {
		success = false;
		std::cout << "Solving failed\n";
	}

	return success;
}



double lin_interp(const double lx, const double lv, const double rx, const double rv, const double x)
{
	return ((rx - x) * lv + (x - lx) * rv) / (rx - lx);
}


double compute_isovalue_3d(
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


double compute_isovalue_2d(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXd& GV,
	const Resolution& res,
	const Eigen::VectorXd& Chi)
{
	double isoval = 0;
	Eigen::Matrix<size_t, 2, 2> sq;
	for (size_t i = 0; i < V.rows(); i++)
	{
		xy2sq(GV, res, V(i, 0), V(i, 1), sq);

		double x_bottom_chi = lin_interp(GV(sq(0, 0), 0), Chi(sq(0, 0)), GV(sq(0, 1), 0), Chi(sq(0, 1)), V(i, 0));
		double x_top_chi = lin_interp(GV(sq(1, 0), 0), Chi(sq(1, 0)), GV(sq(1, 1), 0), Chi(sq(1, 1)), V(i, 0));
		double chi = lin_interp(GV(sq(0, 0), 1), x_bottom_chi, GV(sq(1, 0), 1), x_top_chi, V(i, 1));

		isoval += chi;
	}
	isoval /= V.rows();
	return isoval;
}


double compute_isovalue(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXd& GV,
	const Resolution& res,
	const Eigen::VectorXd& Chi)
{
	if (res.z == 1)
	{
		return compute_isovalue_2d(V, GV, res, Chi);
	}
	else
	{
		return compute_isovalue_3d(V, GV, res, Chi);
	}

}

void compute_normals(const Eigen::MatrixXd& V, Eigen::MatrixXd& N)
{
	N.resize(V.rows(), V.cols());
	double r_max = V.col(0).maxCoeff();
	for (size_t i = 0; i < V.rows(); i++)
	{
		Eigen::RowVector3d v1 = V.row((i + 1) % V.rows()) - V.row(i);
		Eigen::RowVector3d n1;
		n1(0) = 1;
		n1(1) = -v1(0) / v1(1);
		n1(2) = 0;
		n1.normalize();
		if (abs(v1(0)) < 0.000001)
		{
			if (abs(V(i, 0) - r_max) < 0.000001)
				n1 = -n1;
		}
		else
		{
			if (v1(1) < 0)
				n1 = -n1;
		}

		Eigen::RowVector3d v2 = V.row(i) - V.row((i - 1 + V.rows()) % V.rows());
		Eigen::RowVector3d n2;
		n2(0) = 1;
		n2(1) = -v2(0) / v2(1);
		n2(2) = 0;
		n2.normalize();


		if (abs(v2(0)) < 0.000001)
		{
			if (abs(V(i, 0) - r_max) < 0.000001)
				n2 = -n2;
		}
		else
		{
			if (v2(1) < 0)
				n2 = -n2;
		}

		N.row(i) = n1 + n2;
		N.row(i).normalize();

		//if ((i <= 11 && i >= 7) || (i >= 34 && i <= 38))
		//	N.row(i) *= 4;
	}
}



void faces_to_edges(const Eigen::MatrixXi& F, const Eigen::MatrixXd& V, Eigen::MatrixXd& MC_V_start, Eigen::MatrixXd& MC_V_end)
{
	MC_V_start = Eigen::MatrixXd(F.rows() * 3, 3);
	MC_V_end = Eigen::MatrixXd(F.rows() * 3, 3);

	for (int i = 0; i < F.rows(); i++)
	{
		MC_V_start.row(i * 3 + 0) = V.row(F(i, 0));
		MC_V_end.row(i * 3 + 0) = V.row(F(i, 1));

		MC_V_start.row(i * 3 + 1) = V.row(F(i, 1));
		MC_V_end.row(i * 3 + 1) = V.row(F(i, 2));

		MC_V_start.row(i * 3 + 2) = V.row(F(i, 2));
		MC_V_end.row(i * 3 + 2) = V.row(F(i, 0));
	}

}

void modify_chi(const Eigen::MatrixXd& GV, const Eigen::MatrixXd& V_known, const size_t k, const double isovalue, Eigen::VectorXd& X)
{
	// find the grid points near to the vertices that are part of the edges
	// set their chi value to the isovalue
	//size_t k = 4;
	typedef nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixXd >  kd_tree_t;
	kd_tree_t mat_index(GV, 10 /* max leaf */);
	mat_index.index->buildIndex();

	for (int searchPointIdx = 0; searchPointIdx < V_known.rows(); searchPointIdx++) {
		//std::cout << searchPointIdx << "\n";
		Eigen::RowVector3d searchPoint = V_known.row(searchPointIdx);

		// create a query object
		std::vector<size_t> neighbourIdx(k);
		std::vector<double> distsSqr(k);

		nanoflann::KNNResultSet<double> knnSearchResult(k);
		knnSearchResult.init(neighbourIdx.data(), distsSqr.data());

		// find neighbours
		mat_index.index->findNeighbors(knnSearchResult, searchPoint.data(), nanoflann::SearchParams(50));

		for (size_t i = 0; i < neighbourIdx.size(); i++)
		{
			X(neighbourIdx[i]) = isovalue;
		}

	}
}


// this functions is specific to the ext2 shape
void modify_normals(const Eigen::MatrixXd& GV, const double xMin, const double xMax, Eigen::MatrixXd& GN)
{
	// identify x range in which to change normals
	// remove all x component (and then fix back to original magnitude)
	for (int i = 0; i < GV.rows(); i++)
	{
		// check if in range
		if (GV(i, 0) >= xMin && GV(i, 0) <= xMax)
		{
			// if so, modify the associated normal
			//GN(i, 0) = 0.0; // remove horizontal component (should probably try preserving th magnitude)
			GN(i, 0) = -GN(i, 0); // flip horizontal component

		}
	}



}

// add random noise within limits determined by pc (%) of bounding box
void add_noise(const double pointPC, const double normalPC, Eigen::MatrixXd& V, Eigen::MatrixXd& N)
{
	// find bounding area
	Eigen::RowVector3d BBmin = V.colwise().minCoeff();
	Eigen::RowVector3d BBmax = V.colwise().maxCoeff();
	Eigen::RowVector3d BBrange = BBmax - BBmin;
	double BBdiag = BBrange.norm();
	double pointNoiseLevel = BBdiag * pointPC;
	double normalNoiseLevel = BBdiag * normalPC;

	Eigen::MatrixXd pointNoise = Eigen::MatrixXd::Random(V.rows(), 3) * pointNoiseLevel;
	Eigen::MatrixXd normalNoise = Eigen::MatrixXd::Random(V.rows(), 3) * normalNoiseLevel;

	if (BBmin(2) == BBmax(2))	// 2D
	{
		pointNoise.block(0, 2, V.rows(), 1) = Eigen::MatrixXd::Zero(V.rows(), 1);
		normalNoise.block(0, 2, V.rows(), 1) = Eigen::MatrixXd::Zero(V.rows(), 1);
	}

	V += pointNoise;
	N += normalNoise;

}