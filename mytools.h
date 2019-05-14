#ifndef MYTOOLS_H
#define MYTOOLS_H

struct Resolution
{
	size_t x;
	size_t y;
	size_t z;
};


void get_example_mesh(std::string const meshname, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& VN);

// Is there really no way to just get point data only?
void get_point_data(std::string const meshname, Eigen::MatrixXd& V);

void get_point_data(std::string const meshname, Eigen::MatrixXd& V, Eigen::MatrixXd& N);

void get_grid(const Eigen::MatrixXd& V, const int depth, Eigen::MatrixXd& GV, Eigen::MatrixXi& GE, Resolution& res);

void construct_laplacian(const Resolution& res, Eigen::MatrixXd& GV, Eigen::SparseMatrix<double>& L);

void construct_divergence(const Resolution& res,
	Eigen::SparseMatrix<double, Eigen::RowMajor>& Dx,
	Eigen::SparseMatrix<double, Eigen::RowMajor>& Dy);

void conv2d(const Eigen::MatrixXd &M, const Eigen::MatrixXd &K, const Resolution& res, Eigen::MatrixXd &MK);

void compute_grid_normals(
	const Eigen::MatrixXd &V,
	const Eigen::MatrixXd &N,
	const Eigen::MatrixXd &GV,
	const size_t k,
	Eigen::MatrixXd &weightedNormals);

double compute_isovalue(
	const Eigen::MatrixXd &V,
	const Eigen::MatrixXd &GV,
	const Resolution& res,
	const Eigen::VectorXd &Chi);

#endif


