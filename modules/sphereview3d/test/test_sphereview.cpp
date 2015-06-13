#include "../include/sphereview3d.hpp"
using namespace cv;
using namespace std;

int main(int argc, char *argv[]){

	cv::sphereview3d::IcoSphere ViewSphere(16,0);
	std::vector<cv::Point3d>* campos = ViewSphere.CameraPos;
	//IcoSphere ViewSphere(16,0);
	//std::vector<cv::Point3d>* campos = ViewSphere.CameraPos;
	bool camera_pov = (true);
	/// Create a window
	viz::Viz3d myWindow("Coordinate Frame");

	/// Add coordinate axes
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

	/// Let's assume camera has the following properties
	Point3d cam_pos(3.0f,0.0f,0.0f), cam_focal_point(0.0f,0.0f,0.0f), cam_y_dir(-0.0f,-0.0f,-1.0f);
	for(int pose = 0; pose < campos->size(); pose++){
		/// We can get the pose of the cam using makeCameraPose
		Affine3f cam_pose = viz::makeCameraPose(campos->at(pose), cam_focal_point, cam_y_dir);

		/// We can get the transformation matrix from camera coordinate system to global using
		/// - makeTransformToGlobal. We need the axes of the camera
		Affine3f transform = viz::makeTransformToGlobal(Vec3f(0.0f,-1.0f,0.0f), Vec3f(-1.0f,0.0f,0.0f), Vec3f(0.0f,0.0f,-1.0f), campos->at(pose));

		/// Create a cloud widget.

		viz::Mesh objmesh = viz::Mesh::load("./ape.ply");
		viz::WMesh mesh_widget(objmesh);

		/// Pose of the widget in camera frame
		Affine3f cloud_pose = Affine3f().translate(Vec3f(3.0f,3.0f,3.0f));
		/// Pose of the widget in global frame
		Affine3f cloud_pose_global = transform * cloud_pose;

		/// Visualize camera frame
		if (!camera_pov)
		{
			viz::WCameraPosition cpw(0.5); // Coordinate axes
			viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
			myWindow.showWidget("CPW", cpw, cam_pose);
			myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
		}

		/// Visualize widget
		mesh_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
		myWindow.showWidget("bunny", mesh_widget, transform);

		/// Set the viewer pose to that of camera
		if (camera_pov)
			myWindow.setViewerPose(cam_pose);
	}

	/// Start event loop.
	myWindow.spin();
	return 1;
};
