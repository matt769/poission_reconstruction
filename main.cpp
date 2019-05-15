#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/octree.h>
#include <igl/copyleft/marching_cubes.h>
#include <imgui/imgui.h>
#include <iostream>
#include <random>

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
	Eigen::MatrixXd m_GV;
	Eigen::MatrixXd m_V_nn;
	Eigen::MatrixXd m_GN;
	Eigen::MatrixXd m_X;
	Eigen::MatrixXd m_DGV;
	Eigen::MatrixXd m_MC_V;
	Eigen::MatrixXi m_MC_F;

	float nv_len;
	float point_size;
	float line_width;

	MyContext() :nv_len(0), point_size(5), line_width(1)
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
		viewer.data().add_points(m_V, Eigen::RowVector3d(255, 0, 0));

		// now want to show normals
		//m_Nend = m_V + m_N;
		//viewer.data().add_edges(m_V, m_Nend, Eigen::RowVector3d(0, 0, 255));

		// show grid
		//viewer.data().add_points(m_GV, Eigen::RowVector3d(0, 255, 0));

		// show normals at grid points
		m_GNend = m_GV + m_GN;
		viewer.data().add_edges(m_GV, m_GNend, Eigen::RowVector3d(0, 0, 255));

		// show coloured grid points according to indicator values
		Eigen::MatrixXd indicatorColours = Eigen::MatrixXd::Zero(m_GV.rows(), 3);
		indicatorColours.col(2) = m_X - Eigen::VectorXd::Constant(m_X.size(), m_X.minCoeff());
		indicatorColours.col(2) /= indicatorColours.col(2).maxCoeff();
		indicatorColours.col(0) = indicatorColours.col(2);
		viewer.data().add_points(m_GV, indicatorColours);

		// Add marching cube output
		viewer.data().add_points(m_MC_V, Eigen::RowVector3d(0, 255, 0));

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
	// load data  ************************************************************************************
	Eigen::MatrixXd V;
	Eigen::MatrixXd N;
	//get_point_data("circle.obj", V, N);
	get_point_data("sphere.obj", V, N);
	//get_point_data("circle_noisy.obj", V, N);
	//get_point_data("circle_noisy2.obj", V, N);
	//get_point_data("circle_gap.obj", V, N);
	//std::cout << "Sample vertices:" << V.rows() << "\n";
	//std::cout << "V:\n" << V << "\n";


	int depth = 4;
	Eigen::MatrixXd GV;
	Eigen::MatrixXi GE;
	Resolution gridResolution;
	get_grid(V, depth, GV, GE, gridResolution);
	
	//std::cout << "Grid vertices:" << GV.rows() << "\n";
	//std::cout << "GV:\n" << GV << "\n";
	//std::cout << "Grid resolution: " << gridResolution.x << ", " << gridResolution.y << ", " << gridResolution.z << "\n";

	//// interpolate normals to grid points   *******************************************************************
	//// for now, find all within some distance, and treat all equally (unweighted average)
	Eigen::MatrixXd weightedNormals;
	compute_grid_normals(V, N, GV, 8, weightedNormals);
	////std::cout << "Grid normals:" << weightedNormals.rows() << "\n";
	////std::cout << "GN:\n" << weightedNormals << "\n";



	Eigen::MatrixXd VF;
	//Eigen::Matrix3d K;
	//K << 9, 57, 9,
	//	57, 361, 57,
	//	9, 57, 9;
	//K /= K.sum();

	Eigen::Vector3d K;
	K << 9, 57, 9;
	K /= K.sum();


	// Apply (approximation of) gaussian smoother
	//conv2d(weightedNormals, K, gridResolution, VF);

	conv3d(weightedNormals, K, gridResolution, VF);
	weightedNormals = VF;
	////std::cout << "Grid normals after convolution:\n" << weightedNormals << "\n";
	////std::cout << weightedNormals << "\n";


	//// construct Laplacian   ************************************************************************************
	Eigen::SparseMatrix<double> L;
	construct_laplacian(gridResolution, GV, L);
	std::cout << "Laplacian:" << L.rows() << "," << L.cols() << "\n";
	////std::cout << "L:\n" << L << "\n";


	//// construct Divergence   ************************************************************************************
	Eigen::SparseMatrix<double, Eigen::RowMajor> Dx;
	Eigen::SparseMatrix<double, Eigen::RowMajor> Dy;
	Eigen::SparseMatrix<double, Eigen::RowMajor> Dz;
	construct_divergence(gridResolution, Dx, Dy, Dz);
	////std::cout << "Divergence X:" << Dx.rows() << "," << Dx.cols() << "\n";
	////std::cout << "Dx:\n" << Dx << "\n";
	////std::cout << "Divergence Y:" << Dx.rows() << "," << Dy.cols() << "\n";
	////std::cout << "Dy:\n" << Dy << "\n";
	////std::cout << "Divergence Z:" << Dz.rows() << "," << Dz.cols() << "\n";
	////std::cout << "Dz:\n" << Dz << "\n";
	Eigen::MatrixXd DGV = (Dx * weightedNormals.col(0)) + (Dy * weightedNormals.col(1)) + (Dz * weightedNormals.col(2));
	////std::cout << DGV << std::endl;




	//// Solve Poisson equation  ************************************************************************************
	//// Solve system Lx = DGV
	//Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
	Eigen::VectorXd x;

	////std::cout << "L" << '\n';
	////std::cout << L << '\n';
	////std::cout << "DGV" << '\n';
	////std::cout << DGV << '\n';


	solver.compute(L);
	if (solver.info() != Eigen::Success) {
		std::cout << "Decomposition failed\n";
	}
	x = -solver.solve(DGV);
	if (solver.info() != Eigen::Success) {
		std::cout << "Solving failed\n";
	}
	std::cout << "x min, max:\n" << x.minCoeff() << "," << x.maxCoeff() << "\n";

	//// Calculate isovalue that will be used in marching cubes
	//double isoval = compute_isovalue(V, GV, gridResolution, x);
	//std::cout << "isoval: " << isoval << "\n";

	// Run igl marching cubes  ************************************************************************************
	Eigen::MatrixXd MC_V;
	Eigen::MatrixXi MC_F;
	//igl::copyleft::marching_cubes(x, GV, gridResolution.x, gridResolution.y, gridResolution.z, isoval, MC_V, MC_F);
	igl::copyleft::marching_cubes(x, GV, gridResolution.x, gridResolution.y, gridResolution.z, -0.1, MC_V, MC_F);

	
	////std::cout << "MC_V:\n" << MC_V << "\n";
	////std::cout << "MC_V size: " << MC_V.rows() << ", " << MC_V.cols() << "\n";
	//


	//------------------------------------------
	// for visualization
	g_myctx.m_V = V;
	g_myctx.m_N = N;
	g_myctx.m_GV = GV;
	g_myctx.m_GN = weightedNormals;
	g_myctx.m_X = x;
	g_myctx.m_MC_V = MC_V;
	g_myctx.m_MC_F = MC_F;

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

		// number of eigen vectors
		// [event handle] if value changed
		//if (ImGui::InputInt("num_eig", &g_myctx.num_eig))
		//{
		//	std::cout << "num_eig changed\n";
		//	g_myctx.reset_display(viewer);
		//}

		ImGui::End();
	};

	// registered a event handler
	viewer.callback_key_down = &key_down;

	g_myctx.reset_display(viewer);

	// Call GUI
	viewer.launch();

}
