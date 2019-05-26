#ifndef MYTOOLS_H
#define MYTOOLS_H

struct Resolution
{
	int x;
	int y;
	int z;
	Resolution()
	{
		x = 0;
		y = 0;
		z = 0;
	}
	Resolution(int xn, int yn, int zn) {
		x = xn;
		y = yn;
		z = zn;
	}
	Resolution operator +(const Resolution b) {
		return Resolution(x + b.x, y + b.y, z + b.z);
	}

};




void get_example_mesh(std::string const meshname, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& VN);

void get_point_data(std::string const meshname, Eigen::MatrixXd& V);

void get_point_data(std::string const meshname, Eigen::MatrixXd& V, Eigen::MatrixXd& N);



//bool vertex_ij2idx(const Resolution& res, const int xIdx, const int yIdx, size_t& idx);

// Get vertex id from resolution and indices - 3D version
bool vertex_ijk2idx(const Resolution& res, const int xIdx, const int yIdx, const int zIdx, size_t& idx);

bool vertex_ijk2idx(const Resolution& res, const Resolution& xyz, size_t& idx);

bool vertex_idx2ijk(const Resolution& res, const size_t idx, Resolution& xyz);

// Get indices of vertices of containing square
// x and y are position of any point in space
void xy2sq(const Eigen::MatrixXd& GV, const Resolution& res, const double x, const double y, Eigen::Matrix<size_t, 2, 2>& square);

void xyz2cube(const Eigen::MatrixXd& GV, 
				const Resolution& res, 
				const double x, 
				const double y, 
				const double z, 
				std::vector<Eigen::Matrix<size_t, 2, 2>>& cube);



void get_grid(const Eigen::MatrixXd& V, const int depth, const size_t extra_layers, Eigen::MatrixXd& GV, Eigen::MatrixXi& GE, Resolution& res);

void construct_laplacian(const Resolution& res, Eigen::MatrixXd& GV, Eigen::SparseMatrix<double>& L);

void construct_divergence(const Resolution& res,
	Eigen::SparseMatrix<double, Eigen::RowMajor>& Dx,
	Eigen::SparseMatrix<double, Eigen::RowMajor>& Dy,
	Eigen::SparseMatrix<double, Eigen::RowMajor>& Dz);

void apply_convolution(const Eigen::MatrixXd& M, const Eigen::VectorXd& K, const Resolution& res, Eigen::MatrixXd& MK);

void compute_grid_normals(
	const Eigen::MatrixXd &V,
	const Eigen::MatrixXd &N,
	const Eigen::MatrixXd &GV,
	const size_t k,
	Eigen::MatrixXd &weightedNormals);

void compute_grid_normals(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXd& N,
	const Eigen::MatrixXd& GV,
	const Resolution& res,
	Eigen::MatrixXd& GN);

bool solve_poisson_equation(const Eigen::SparseMatrix<double> L, const Eigen::MatrixXd DGV, Eigen::VectorXd& x);

double compute_isovalue_3d(
	const Eigen::MatrixXd &V,
	const Eigen::MatrixXd &GV,
	const Resolution& res,
	const Eigen::VectorXd &Chi);

double compute_isovalue_2d(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXd& GV,
	const Resolution& res,
	const Eigen::VectorXd& Chi);

double compute_isovalue(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXd& GV,
	const Resolution& res,
	const Eigen::VectorXd& Chi);

void compute_normals(const Eigen::MatrixXd& V, Eigen::MatrixXd& N);

void faces_to_edges(const Eigen::MatrixXi& F, const Eigen::MatrixXd& V, Eigen::MatrixXd& MC_V_start, Eigen::MatrixXd& MC_V_end);

void modify_chi(const Eigen::MatrixXd& GV, const Eigen::MatrixXd& V_known, const size_t k, const double isovalue, Eigen::VectorXd& X);

void modify_normals(const Eigen::MatrixXd& GV, const double xMin, const double xMax, Eigen::MatrixXd& GN);

void add_noise(const double pointPC, const double normalPC, Eigen::MatrixXd& V, Eigen::MatrixXd& N);


#endif


