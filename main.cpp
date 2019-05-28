#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/octree.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/jet.h>
#include <imgui/imgui.h>
#include <iostream>
#include <random>
//#include <ctime>
#include <string>

#include "nanoflann.hpp"
#include "mytools.h"



class MyContext
{
public:

	Eigen::MatrixXd m_V;
	Eigen::MatrixXi m_F;	 
	Eigen::MatrixXd m_N;
	Eigen::MatrixXd m_Nend;
	Eigen::MatrixXd m_GNend;
	Eigen::MatrixXd m_GN_smoothed_end;
	Eigen::MatrixXd m_GV;
	Eigen::MatrixXd m_V_nn;
	Eigen::MatrixXd m_GN;
	Eigen::MatrixXd m_GN_smoothed;
	Eigen::MatrixXd m_X;
	Eigen::MatrixXd m_DGV;
	Eigen::MatrixXd m_MC_V;
	Eigen::MatrixXi m_MC_F;
	Eigen::MatrixXd m_MC_E_start;
	Eigen::MatrixXd m_MC_E_end;

	Resolution res;

	float nv_len;
	float point_size;
	float line_width;
	int layer_no;
	bool enable_MC_faces;
	bool enable_sample_normals;
	bool enable_sample_points;
	bool enable_grid_normals;
	bool enable_smoothed_grid_normals;

	MyContext() :nv_len(0), point_size(5), line_width(1), layer_no(-1), 
					enable_MC_faces(false), enable_sample_normals(false), enable_grid_normals(false), 
					enable_smoothed_grid_normals(false), enable_sample_points(true)
	{

	}
	~MyContext() {}

	void reset_display(igl::opengl::glfw::Viewer& viewer)
	{
		
		viewer.data().clear(); 
		// hide default wireframe
		viewer.data().show_lines = 0;
		viewer.data().show_overlay_depth = 1; 


		//======================================================================

		viewer.data().line_width = line_width;
		viewer.data().point_size = point_size;

		viewer.data().clear();

		// show sample points
		if (enable_sample_points)
		{
			viewer.data().add_points(m_V, Eigen::RowVector3d(0, 0, 0));
		}


		// show sample normals
		if (enable_sample_normals)
		{
			m_Nend = m_V + m_N;
			viewer.data().add_edges(m_V, m_Nend, Eigen::RowVector3d(0, 0, 255));
		}


		// show grid
		//viewer.data().add_points(m_GV, Eigen::RowVector3d(0, 255, 0));
		
		Eigen::MatrixXd indicatorColours;
		igl::jet(m_X, true, indicatorColours);

		switch (layer_no)
		{
		case -3:
			layer_no = -2;
		case -2:
			break;
		case -1:
			viewer.data().add_points(m_GV, indicatorColours);
			break;
		default:
			if (layer_no < res.z)
			{

				viewer.data().add_points(m_GV.block(res.x * res.y * layer_no, 0, res.x * res.y, 3),
					indicatorColours.block(res.x * res.y * layer_no, 0, res.x * res.y, 3));
			}
			else {
				layer_no = res.z - 1;
			}
			break;
		}


		// show normals at grid points
		if (enable_grid_normals)
		{
			m_GNend = m_GV + m_GN;
			viewer.data().add_edges(m_GV, m_GNend, Eigen::RowVector3d(0, 0, 255));
		}

		// show smoothed normals at grid points
		if (enable_smoothed_grid_normals)
		{
			m_GN_smoothed_end = m_GV + m_GN_smoothed;
			viewer.data().add_edges(m_GV, m_GN_smoothed_end, Eigen::RowVector3d(0, 0, 255));
		}


		// Add marching cube output
		//viewer.data().add_points(m_MC_V, Eigen::RowVector3d(0, 255, 0));
		if (enable_MC_faces)
		{
			if (res.z == 1)
			{
				faces_to_edges(m_MC_F, m_MC_V, m_MC_E_start, m_MC_E_end);
				viewer.data().add_points(m_MC_V, Eigen::RowVector3d(0, 255, 0));
				viewer.data().add_edges(m_MC_E_start, m_MC_E_end, Eigen::RowVector3d(0, 255, 0));
			}
			else
			{
				viewer.data().set_mesh(m_MC_V, m_MC_F);
			}
		}
		
	}

private:

};

MyContext g_myctx;


bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{

	std::cout << "Key: " << key << " " << (unsigned int)key << std::endl;
	if (key=='q' || key=='Q')
	{
		exit(0);
	}
	return false;
}



int main(int argc, char *argv[])
{

	//std::cout << "You have entered " << argc
	//	<< " arguments:" << "\n";

	//for (int i = 0; i < argc; ++i)
	//	std::cout << argv[i] << "\n";


	time_t timeStart;
	timeStart = std::time(NULL);
	std::cout << "Start time:" << timeStart << "\n";

	// default parameters ************************************************************************************
	int depth = 5;
	size_t extra_layers = 3;
	double pointPC = 0.0;
	double normalPC = 0.0;
	double isovalue = 0.0;
	bool customIsovalue = false;


	// check command line args ***********************************************************************
	//for (int i = 0; i < argc; ++i)
	//	std::cout << argv[i] << "\n";


	std::string fileName;
	if (argc > 1)
	{
		fileName = std::string(argv[1]);
	}
	else
	{
		fileName = "circle.obj"; // CHANGE FILENAME HERE IF NOT PROVIDING VIA COMMAND LINE
	}

	if (argc > 2) depth = atoi(argv[2]);
	if (argc > 3) extra_layers = atoi(argv[3]);
	if (argc > 4) pointPC = atof(argv[4]);
	if (argc > 5) normalPC = atof(argv[5]);
	if (argc > 6)
	{
		isovalue = atof(argv[6]);
		customIsovalue = true;
	}


	std::cout << "Depth:" << depth << "\n";
	std::cout << "Extra layers:" << extra_layers << "\n";

	// load data  ************************************************************************************
	std::cout << "Loading point data\n";
	Eigen::MatrixXd V;
	Eigen::MatrixXd N;
	Eigen::MatrixXi F;

	if (fileName == "bunny.obj")
	{
		get_example_mesh(fileName, V, F, N);
		N = -N; // Need to fix the normals to be pointing inside
		N /= 5; // And make them a little smaller too (just for viewing, shouldn't affect the result)
	}
	else if (fileName == "ext2.obj")
	{
		// EXTENSION MODIFICATION
		get_point_data(fileName, V);
		compute_normals(V, N);
	}
	else
	{
		get_point_data(fileName, V, N);
	}


	// add noise **************************************************************************************
	if (pointPC > 0 || normalPC > 0)
	{
		add_noise(pointPC, normalPC, V, N);
	}




	//std::cout << "Sample vertices:" << V.rows() << "\n";
	//std::cout << "V:\n" << V << "\n";

	// create grid ************************************************************************************
	std::cout << "Creating grid\n";
	Eigen::MatrixXd GV;
	Eigen::MatrixXi GE;
	Resolution gridResolution;
	get_grid(V, depth, extra_layers, GV, GE, gridResolution);
	
	//std::cout << "Grid vertices:" << GV.rows() << "\n";
	//std::cout << "GV:\n" << GV << "\n";
	//std::cout << "Grid resolution: " << gridResolution.x << ", " << gridResolution.y << ", " << gridResolution.z << "\n";

	// interpolate normals to grid points   *******************************************************************
	std::cout << "Spread normals to grid points\n";
	Eigen::MatrixXd GN; // ** TODO ** initialise this to zeros
	//size_t numGridPoints; // how many grid points to spread each sample point to
	//if (gridResolution.z == 1) numGridPoints = 4;
	//else numGridPoints = 8;
	//compute_grid_normals(V, N, GV, numGridPoints, GN);
	compute_grid_normals(V, N, GV, gridResolution, GN);
	//std::cout << "Grid normals:" << GN.rows() << "\n";
	//std::cout << "GN:\n" << GN << "\n";


	// Apply smoothing filter to normals   *******************************************************************
	std::cout << "Applying smoothing filter\n";
	Eigen::MatrixXd GN_smoothed;
	Eigen::Vector3d K;
	K << 9, 57, 9;
	//Eigen::Matrix<double, 5, 1> K;
	//K << 7, 26, 41, 26, 7;
	//Eigen::Matrix<double, 5, 1> K;
	//K << 1, 1, 1, 1, 1;
	//Eigen::Matrix<double, 7, 1> K;
	//K << 1, 1, 1, 1, 1, 1, 1;
	/*Eigen::Matrix<double, 9, 1> K;
	K << 1, 1, 1, 1, 1, 1, 1, 1, 1;*/
	//Eigen::Matrix<double, 1, 1> K;
	//K << 1;
	K /= K.sum();
	// *** It doesn't seems to make any visual difference to the result


	// Apply (approximation of) gaussian smoother
	apply_convolution(GN, K, gridResolution, GN_smoothed);
	
	//GN = GN_smoothed;
	////std::cout << "Grid normals after convolution:\n" << weightedNormals << "\n";
	////std::cout << weightedNormals << "\n";

	// EXTENSION MODIFICATION
	//double xMin = -0.2;
	//double xMax = 0.2;
	//modify_normals(GV, xMin, xMax, GN_smoothed);



	//// construct Laplacian   ************************************************************************************
	std::cout << "Constructing Laplacian\n";
	Eigen::SparseMatrix<double> L;
	construct_laplacian(gridResolution, GV, L);
	//std::cout << "Laplacian:" << L.rows() << "," << L.cols() << "\n";
	//std::cout << "L:\n" << L << "\n";


	//// construct Divergence   ************************************************************************************
	std::cout << "Constructing divergence\n";
	Eigen::SparseMatrix<double, Eigen::RowMajor> Dx;
	Eigen::SparseMatrix<double, Eigen::RowMajor> Dy;
	Eigen::SparseMatrix<double, Eigen::RowMajor> Dz;
	construct_divergence(gridResolution, Dx, Dy, Dz);
	//std::cout << "Divergence X:" << Dx.rows() << "," << Dx.cols() << "\n";
	//std::cout << "Dx:\n" << Dx << "\n";
	//std::cout << "Divergence Y:" << Dx.rows() << "," << Dy.cols() << "\n";
	//std::cout << "Dy:\n" << Dy << "\n";
	//std::cout << "Divergence Z:" << Dz.rows() << "," << Dz.cols() << "\n";
	//std::cout << "Dz:\n" << Dz << "\n";
	Eigen::MatrixXd DGV = (Dx * GN_smoothed.col(0)) + (Dy * GN_smoothed.col(1)) + (Dz * GN_smoothed.col(2));
	//std::cout << "Divergence:" << DGV.rows() << "," << DGV.cols() << "\n";
	//std::cout << DGV << std::endl;




	//// Solve Poisson equation  ************************************************************************************
	std::cout << "Solving Poisson equation\n";
	//// Solve system Lx = DGV
	Eigen::VectorXd x;
	solve_poisson_equation(L, DGV, x);

	std::cout << "x min, max:\n" << x.minCoeff() << "," << x.maxCoeff() << "\n";

	//// Calculate isovalue that will be used in marching cubes
	if (!customIsovalue)
	{
		std::cout << "Computing isovalue\n";
		isovalue = compute_isovalue(V, GV, gridResolution, x);
	}
	std::cout << "Isovalue: " << isovalue << "\n";

	//isoval = -0.3;

	// Run igl marching cubes  ************************************************************************************
	std::cout << "Running marching cubes\n";
	Eigen::MatrixXd MC_V;
	Eigen::MatrixXi MC_F;

	if (gridResolution.z == 1)
	{
		Eigen::VectorXd x_3d(2 * x.rows());
		x_3d.block(0, 0, x.rows(), 1) = x;
		x_3d.block(x.rows(), 0, x.rows(), 1) = Eigen::VectorXd::Constant(x.rows(), x.maxCoeff());
		Eigen::MatrixXd GV_3d(2 * GV.rows(), GV.cols());
		GV_3d.block(0, 0, GV.rows(), GV.cols()) = GV;
		GV_3d.block(GV.rows(), 0, GV.rows(), GV.cols()) = GV;

		igl::copyleft::marching_cubes(x_3d, GV_3d, gridResolution.x, gridResolution.y, gridResolution.z+1, isovalue, MC_V, MC_F);
	}
	else
	{
		igl::copyleft::marching_cubes(x, GV, gridResolution.x, gridResolution.y, gridResolution.z, isovalue, MC_V, MC_F);
	}

	//std::cout << "MC_V:\n" << MC_V << "\n";
	//std::cout << "MC_F:\n" << MC_F << "\n";
	//std::cout << "MC_V size: " << MC_V.rows() << ", " << MC_V.cols() << "\n";
	
	// EXTENSION MODIFICATION
	// Try modifying the chi values of points near the known edges
	//Eigen::MatrixXd V_known(5, 3);
	////Eigen::MatrixXd V_known(10, 3);
	//V_known.block(0, 0, 5, 3) = V.block(7, 0, 5, 3);
	////V_known.block(5, 0, 5, 3) = V.block(34, 0, 5, 3);
	//size_t k = 2;
	//modify_chi(GV, V_known, k, isoval, x);








	std::cout << "Finished calculations\n";
	time_t timeEnd;
	timeEnd = std::time(NULL);
	std::cout << "End time:" << timeEnd << "\n";
	std::cout << "Elapsed time:" << timeEnd - timeStart << "\n";

	//------------------------------------------
	// for visualization
	g_myctx.m_V = V;
	g_myctx.m_N = N;
	g_myctx.m_GV = GV;
	g_myctx.m_GN = GN;
	g_myctx.m_GN_smoothed = GN_smoothed;
	g_myctx.m_X = x;
	g_myctx.m_MC_V = MC_V;
	g_myctx.m_MC_F = MC_F;
	g_myctx.res = gridResolution;

	//------------------------------------------
	// Init the viewer
	igl::opengl::glfw::Viewer viewer;

	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	// menu variable Shared between two menus
	double doubleVariable = 0.1f; 

	// Add content to the default menu window via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("New Group", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Expose variable directly ...
			ImGui::InputDouble("double", &doubleVariable, 0, 0, "%.4f");

			// ... or using a custom callback
			static bool boolVariable = true;
			if (ImGui::Checkbox("bool", &boolVariable))
			{
				// do something
				std::cout << "boolVariable: " << std::boolalpha << boolVariable << std::endl;
			}

			// Expose an enumeration type
			enum Orientation { Up = 0, Down, Left, Right };
			static Orientation dir = Up;
			ImGui::Combo("Direction", (int *)(&dir), "Up\0Down\0Left\0Right\0\0");

			// We can also use a std::vector<std::string> defined dynamically
			static int num_choices = 3;
			static std::vector<std::string> choices;
			static int idx_choice = 0;
			if (ImGui::InputInt("Num letters", &num_choices))
			{
				num_choices = std::max(1, std::min(26, num_choices));
			}
			if (num_choices != (int)choices.size())
			{
				choices.resize(num_choices);
				for (int i = 0; i < num_choices; ++i)
					choices[i] = std::string(1, 'A' + i);
				if (idx_choice >= num_choices)
					idx_choice = num_choices - 1;
			}
			ImGui::Combo("Letter", &idx_choice, choices);

		}
	};

	// Add additional windows via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_custom_window = [&]()
	{
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(250, 400), ImGuiSetCond_FirstUseEver);
		ImGui::Begin( "MyProperties", nullptr, ImGuiWindowFlags_NoSavedSettings );
		
		// point size
		// [event handle] if value changed
		if (ImGui::InputFloat("point_size", &g_myctx.point_size))
		{
			std::cout << "point_size changed\n";
			viewer.data().point_size = g_myctx.point_size;
		}

		// line width
		// [event handle] if value changed
		if(ImGui::InputFloat("line_width", &g_myctx.line_width))
		{
			std::cout << "line_width changed\n";
			viewer.data().line_width = g_myctx.line_width;
		}

		// layer of grid being displayed
		if (ImGui::InputInt("layer number", &g_myctx.layer_no))
		{
			std::cout << "layer number changed\n";
			g_myctx.reset_display(viewer);
		}

		if (ImGui::Checkbox("Enable MC faces", &g_myctx.enable_MC_faces))
		{
			std::cout << "MC face option changed\n";
			g_myctx.reset_display(viewer);
		}

		
		if (ImGui::Checkbox("Enable sample points", &g_myctx.enable_sample_points))
		{
			std::cout << "Sample points option changed\n";
			g_myctx.reset_display(viewer);
		}

		if (ImGui::Checkbox("Enable sample normals", &g_myctx.enable_sample_normals))
		{
			std::cout << "Sample normals option changed\n";
			g_myctx.reset_display(viewer);
		}

		if (ImGui::Checkbox("Enable grid normals", &g_myctx.enable_grid_normals))
		{
			std::cout << "Grid normals option changed\n";
			g_myctx.reset_display(viewer);
		}

		if (ImGui::Checkbox("Enable smoothed grid normals", &g_myctx.enable_smoothed_grid_normals))
		{
			std::cout << "Smoothed grid normals option changed\n";
			g_myctx.reset_display(viewer);
		}

		ImGui::End();
	};

	// registered a event handler
	viewer.callback_key_down = &key_down;

	g_myctx.reset_display(viewer);

	// Call GUI
	viewer.launch();

}
