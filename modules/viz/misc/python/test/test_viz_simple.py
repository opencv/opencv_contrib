import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

def generate_test_trajectory():
    result =  []
    angle_i = np.arange(0, 271, 3)
    angle_j = np.arange(0, 1200, 10)
    for i, j in zip(angle_i, angle_j):
        x = 2 * np.cos(i * 3 * np.pi/180.0) * (1.0 + 0.5 * np.cos(1.2 + i * 1.2 * np.pi/180.0))
        y = 0.25 + i/270.0 + np.sin(j * np.pi/180.0) * 0.2 * np.sin(0.6 + j * 1.5 * np.pi/180.0)
        z = 2 * np.sin(i * 3 * np.pi/180.0) * (1.0 + 0.5 * np.cos(1.2 + i * np.pi/180.0))
        result.append(cv.viz.makeCameraPose((x, y, z), (0.0, 0, 0), (0.0, 1.0, 0.0)))
    x =  np.zeros(shape=(len(result), 1, 16 ), dtype= np.float64)
    for idx, m in enumerate(result):
        x[idx, 0, :] = m.mat().reshape(16)
    return x, result

def tutorial3(camera_pov, filename):
    myWindow = cv.viz_Viz3d("Coordinate Frame")
    myWindow.showWidget("axe",cv.viz_WCoordinateSystem())

    cam_origin =  (3.0, 3.0, 3.0)
    cam_focal_point = (3.0,3.0,2.0)
    cam_y_dir = (-1.0,0.0,0.0)
    camera_pose = cv.viz.makeCameraPose(cam_origin, cam_focal_point, cam_y_dir)
    transform = cv.viz.makeTransformToGlobal((0.0,-1.0,0.0), (-1.0,0.0,0.0), (0.0,0.0,-1.0), cam_origin)
    dragon_cloud,_,_ = cv.viz.readCloud(filename)
    cloud_widget = cv.viz_WCloud(dragon_cloud, cv.viz_Color().green())
    cloud_pose = cv.viz_Affine3d()
    cloud_pose = cv.viz_Affine3d().rotate((0, np.pi / 2, 0)).translate((0, 0, 3))
    cloud_pose_global = transform.product(cloud_pose)
    myWindow.showWidget("CPW_FRUSTUM", cv.viz_WCameraPosition((0.889484, 0.523599)), camera_pose)
    if not camera_pov:
        myWindow.showWidget("CPW", cv.viz_WCameraPosition(0.5), camera_pose)
    myWindow.showWidget("dragon", cloud_widget, cloud_pose_global)
    if camera_pov:
        myWindow.setViewerPose(camera_pose)

class viz_test(NewOpenCVTests):
    def setUp(self):
        super(viz_test, self).setUp()
        if not bool(os.environ.get('OPENCV_PYTEST_RUN_VIZ', False)):
            self.skipTest("Use OPENCV_PYTEST_RUN_VIZ=1 to enable VIZ UI tests")

    def test_viz_tutorial3_global_view(self):
        tutorial3(False, self.find_file("viz/dragon.ply"))

    def test_viz_tutorial3_camera_view(self):
        tutorial3(True, self.find_file("viz/dragon.ply"))

    def test_viz(self):
        dragon_cloud,_,_ = cv.viz.readCloud(self.find_file("viz/dragon.ply"))
        myWindow = cv.viz_Viz3d("abc")
        myWindow.showWidget("coo", cv.viz_WCoordinateSystem(1))
        myWindow.showWidget("cloud", cv.viz_WPaintedCloud(dragon_cloud))
        myWindow.spinOnce(500, True)

    def test_viz_show_simple_widgets(self):
        viz = cv.viz_Viz3d("show_simple_widgets")
        viz.setBackgroundMeshLab()

        viz.showWidget("coos", cv.viz_WCoordinateSystem())
        viz.showWidget("cube", cv.viz_WCube())
        viz.showWidget("cub0", cv.viz_WCube((-1.0, -1, -1), (-0.5, -0.5, -0.5), False, cv.viz_Color().indigo()))
        viz.showWidget("arro", cv.viz_WArrow((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), 0.009, cv.viz_Color().raspberry()))
        viz.showWidget("cir1", cv.viz_WCircle(0.5, 0.01, cv.viz_Color.bluberry()))
        viz.showWidget("cir2", cv.viz_WCircle(0.5, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), 0.01, cv.viz_Color().apricot()))

        viz.showWidget("cyl0", cv.viz_WCylinder((-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), 0.125, 30, cv.viz_Color().brown()))
        viz.showWidget("con0", cv.viz_WCone(0.25, 0.125, 6, cv.viz_Color().azure()))
        viz.showWidget("con1", cv.viz_WCone(0.125, (0.5, -0.5, 0.5), (0.5, -1.0, 0.5), 6, cv.viz_Color().turquoise()))
        text2d = cv.viz_WText("Different simple widgets", (20, 20), 20, cv.viz_Color().green())
        viz.showWidget("text2d", text2d)
        text3d = cv.viz_WText3D("Simple 3D text", ( 0.5,  0.5, 0.5), 0.125, False, cv.viz_Color().green())
        viz.showWidget("text3d", text3d)

        viz.showWidget("plane1", cv.viz_WPlane((0.25, 0.75)))
        viz.showWidget("plane2", cv.viz_WPlane((0.5, -0.5, -0.5), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.5), cv.viz_Color().gold()))

        viz.showWidget("grid1", cv.viz_WGrid((7,7), (0.75,0.75), cv.viz_Color().gray()), cv.viz_Affine3d().translate((0.0, 0.0, -1.0)))

        viz.spinOnce(500, True)
        text2d.setText("Different simple widgets (updated)")
        text3d.setText("Updated text 3D")
        viz.spinOnce(500, True)

    def test_viz_show_overlay_image(self):
        lena = cv.imread(self.find_file("viz/lena.png"))
        gray = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)
        rows = lena.shape[0]
        cols = lena.shape[1]
        half_lsize = (lena.shape[1] // 2, lena.shape[0] // 2)

        viz = cv.viz_Viz3d("show_overlay_image")
        viz.setBackgroundMeshLab();
        vsz = viz.getWindowSize()

        viz.showWidget("coos", cv.viz_WCoordinateSystem())
        viz.showWidget("cube", cv.viz_WCube())
        x = cv.viz_WImageOverlay(lena, (10, 10, half_lsize[1], half_lsize[0]))
        viz.showWidget("img1", x)
        viz.showWidget("img2", cv.viz_WImageOverlay(gray, (vsz[0] - 10 - cols // 2, 10, half_lsize[1], half_lsize[0])))
        viz.showWidget("img3", cv.viz_WImageOverlay(gray, (10, vsz[1] - 10 - rows // 2, half_lsize[1], half_lsize[0])))
        viz.showWidget("img5", cv.viz_WImageOverlay(lena, (vsz[0] - 10 - cols // 2, vsz[1] - 10 -  rows // 2, half_lsize[1], half_lsize[0])))
        viz.showWidget("text2d", cv.viz_WText("Overlay images", (20, 20), 20, cv.viz_Color().green()))

        i = 0
        for num in range(50):
            i = i + 1
            a = i % 360
            pose = (3 * np.sin(a * np.pi/180), 2.1, 3 * np.cos(a * np.pi/180));
            viz.setViewerPose(cv.viz.makeCameraPose(pose , (0.0, 0.5, 0.0), (0.0, 0.1, 0.0)))
            img = lena * (np.sin(i * 10 * np.pi/180) * 0.5 + 0.5)
            x.setImage(img.astype(np.uint8))
            viz.spinOnce(100, True)
        viz.showWidget("text2d", cv.viz_WText("Overlay images (stopped)", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)

    def test_viz_show_image_3d(self):
        lena = cv.imread(self.find_file("viz/lena.png"))
        lena_gray = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

        viz = cv.viz_Viz3d("show_image_3d")
        viz.setBackgroundMeshLab()
        viz.showWidget("coos", cv.viz_WCoordinateSystem())
        viz.showWidget("cube", cv.viz_WCube());
        viz.showWidget("arr0", cv.viz_WArrow((0.5, 0.0, 0.0), (1.5, 0.0, 0.0), 0.009, cv.viz_Color().raspberry()))
        x = cv.viz_WImage3D(lena, (1.0, 1.0))
        viz.showWidget("img0", x, cv.viz_Affine3d((0.0, np.pi/2, 0.0), (.5, 0.0, 0.0)))
        viz.showWidget("arr1", cv.viz_WArrow((-0.5, -0.5, 0.0), (0.2, 0.2, 0.0), 0.009, cv.viz_Color().raspberry()))
        viz.showWidget("img1", cv.viz_WImage3D(lena_gray, (1.0, 1.0), (-0.5, -0.5, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))

        viz.showWidget("arr3", cv.viz_WArrow((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), 0.009, cv.viz_Color().raspberry()))

        viz.showWidget("text2d", cv.viz_WText("Images in 3D", (20, 20), 20, cv.viz_Color().green()))

        i = 0
        for num in range(50):
            img = lena * (np.sin(i*7.5*np.pi/180) * 0.5 + 0.5)
            x.setImage(img.astype(np.uint8))
            i = i + 1
            viz.spinOnce(100, True);
        viz.showWidget("text2d", cv.viz_WText("Images in 3D (stopped)", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)



    def test_viz_show_cloud_bluberry(self):
        dragon_cloud,_,_ = cv.viz.readCloud(self.find_file("viz/dragon.ply"))

        pose = cv.viz_Affine3d()
        pose = pose.rotate((0, 0.8, 0));
        viz = cv.viz_Viz3d("show_cloud_bluberry")
        viz.setBackgroundColor(cv.viz_Color().black())
        viz.showWidget("coosys", cv.viz_WCoordinateSystem())
        viz.showWidget("dragon", cv.viz_WCloud(dragon_cloud, cv.viz_Color().bluberry()), pose)

        viz.showWidget("text2d", cv.viz_WText("Bluberry cloud", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)

    def test_viz_show_cloud_random_color(self):
        dragon_cloud,_,_ = cv.viz.readCloud(self.find_file("viz/dragon.ply"))

        colors = np.random.randint(0, 255, size=(dragon_cloud.shape[0],dragon_cloud.shape[1],3), dtype=np.uint8)

        pose = cv.viz_Affine3d()
        pose = pose.rotate((0, 0.8, 0));

        viz = cv.viz_Viz3d("show_cloud_random_color")
        viz.setBackgroundMeshLab()
        viz.showWidget("coosys", cv.viz_WCoordinateSystem())
        viz.showWidget("dragon", cv.viz_WCloud(dragon_cloud, colors), pose)
        viz.showWidget("text2d", cv.viz_WText("Random color cloud", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)

    def test_viz_show_cloud_masked(self):
        dragon_cloud,_,_  = cv.viz.readCloud(self.find_file("viz/dragon.ply"))

        qnan =  np.NAN
        for idx in range(dragon_cloud.shape[0]):
            if idx % 15 != 0:
                dragon_cloud[idx,:] = qnan

        pose = cv.viz_Affine3d()
        pose = pose.rotate((0, 0.8, 0))


        viz = cv.viz_Viz3d("show_cloud_masked");
        viz.showWidget("coosys", cv.viz_WCoordinateSystem())
        viz.showWidget("dragon", cv.viz_WCloud(dragon_cloud), pose)
        viz.showWidget("text2d", cv.viz_WText("Nan masked cloud", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)

    def test_viz_show_cloud_collection(self):
        cloud,_,_  = cv.viz.readCloud(self.find_file("viz/dragon.ply"))
        ccol = cv.viz_WCloudCollection()
        pose = cv.viz_Affine3d()
        pose1 =  cv.viz_Affine3d().translate((0, 0, 0)).rotate((np.pi/2, 0, 0))
        ccol.addCloud(cloud, cv.viz_Color().white(), cv.viz_Affine3d().translate((0, 0, 0)).rotate((np.pi/2, 0, 0)))
        ccol.addCloud(cloud, cv.viz_Color().blue(), cv.viz_Affine3d().translate((1, 0, 0)))
        ccol.addCloud(cloud, cv.viz_Color().red(), cv.viz_Affine3d().translate((2, 0, 0)))
        ccol.finalize();

        viz = cv.viz_Viz3d("show_cloud_collection")
        viz.setBackgroundColor(cv.viz_Color().mlab())
        viz.showWidget("coosys", cv.viz_WCoordinateSystem());
        viz.showWidget("ccol", ccol);
        viz.showWidget("text2d", cv.viz_WText("Cloud collection", (20, 20), 20, cv.viz_Color(0, 255,0 )))
        viz.spinOnce(500, True)

    def test_viz_show_painted_clouds(self):
        cloud,_,_  = cv.viz.readCloud(self.find_file("viz/dragon.ply"))
        viz = cv.viz_Viz3d("show_painted_clouds")
        viz.setBackgroundMeshLab()
        viz.showWidget("coosys", cv.viz_WCoordinateSystem())
        pose1 = cv.viz_Affine3d((0.0, -np.pi/2, 0.0), (-1.5, 0.0, 0.0))
        pose2 = cv.viz_Affine3d((0.0, np.pi/2, 0.0), (1.5, 0.0, 0.0))

        viz.showWidget("cloud1", cv.viz_WPaintedCloud(cloud), pose1)
        viz.showWidget("cloud2", cv.viz_WPaintedCloud(cloud, (0.0, -0.75, -1.0), (0.0, 0.75, 0.0)), pose2);
        viz.showWidget("cloud3", cv.viz_WPaintedCloud(cloud, (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), cv.viz_Color().blue(), cv.viz_Color().red()))
        viz.showWidget("arrow", cv.viz_WArrow((0.0, 1.0, -1.0), (0.0, 1.0, 1.0), 0.009, cv.viz_Color()))
        viz.showWidget("text2d", cv.viz_WText("Painted clouds", (20, 20), 20, cv.viz_Color(0, 255, 0)))
        viz.spinOnce(500, True)

    def test_viz_show_mesh(self):
        mesh  = cv.viz.readMesh(self.find_file("viz/dragon.ply"))

        viz = cv.viz_Viz3d("show_mesh")
        viz.showWidget("coosys", cv.viz_WCoordinateSystem());
        viz.showWidget("mesh", cv.viz_WMesh(mesh), cv.viz_Affine3d().rotate((0, 0.8, 0)));
        viz.showWidget("text2d", cv.viz_WText("Just mesh", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)


    def test_viz_show_mesh_random_colors(self):
        mesh  = cv.viz.readMesh(self.find_file("viz/dragon.ply"))
        mesh.colors = np.random.randint(0, 255, size=mesh.colors.shape, dtype=np.uint8)
        viz = cv.viz_Viz3d("show_mesh")
        viz.showWidget("coosys", cv.viz_WCoordinateSystem());
        viz.showWidget("mesh", cv.viz_WMesh(mesh), cv.viz_Affine3d().rotate((0, 0.8, 0)))
        viz.setRenderingProperty("mesh", cv.viz.SHADING, cv.viz.SHADING_PHONG)
        viz.showWidget("text2d", cv.viz_WText("Random color mesh", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)

    def test_viz_show_textured_mesh(self):
        lena = cv.imread(self.find_file("viz/lena.png"))

        angle =  np.arange(0,64)
        points0 = np.vstack((np.zeros(shape=angle.shape, dtype=np.float32), np.cos(angle * np.pi /128), np.sin(angle* np.pi /128)))
        points1 = np.vstack((1.57 * np.ones(shape=angle.shape, dtype=np.float32),np.cos(angle* np.pi /128), np.sin(angle* np.pi /128)))
        tcoords0 = np.vstack((np.zeros(shape=angle.shape, dtype=np.float32), angle / 64))
        tcoords1 = np.vstack((np.ones(shape=angle.shape, dtype=np.float32), angle / 64))
        points =  np.zeros(shape=(points0.shape[0], points0.shape[1] * 2 ),dtype=np.float32)
        tcoords =  np.zeros(shape=(tcoords0.shape[0], tcoords0.shape[1] * 2),dtype=np.float32)
        tcoords[:,0::2] = tcoords0
        tcoords[:,1::2] = tcoords1
        points[:,0::2] = points0 * 0.75
        points[:,1::2] = points1 * 0.75
        polygons =  np.zeros(shape=(4 * (points.shape[1]-2)+1),dtype=np.int32)
        for idx in range(points.shape[1] // 2 - 1):
            polygons[8 * idx: 8 * (idx + 1)] = [3, 2*idx, 2*idx+1, 2*idx+2, 3, 2*idx+1, 2*idx+2, 2*idx+3]

        mesh = cv.viz_Mesh()
        mesh.cloud = points.transpose().reshape(1,points.shape[1],points.shape[0])
        mesh.tcoords = tcoords.transpose().reshape(1,tcoords.shape[1],tcoords.shape[0])
        mesh.polygons = polygons.reshape(1, 4 * (points.shape[1]-2)+1)
        mesh.texture = lena
        viz = cv.viz_Viz3d("show_textured_mesh")
        viz.setBackgroundMeshLab();
        viz.showWidget("coosys", cv.viz_WCoordinateSystem());
        viz.showWidget("mesh", cv.viz_WMesh(mesh))
        viz.setRenderingProperty("mesh", cv.viz.SHADING, cv.viz.SHADING_PHONG)
        viz.showWidget("text2d", cv.viz_WText("Textured mesh", (20, 20), 20, cv.viz_Color().green()));
        viz.spinOnce(500, True)

    def test_viz_show_polyline(self):
        palette = [ cv.viz_Color().red(),
                    cv.viz_Color().green(),
                    cv.viz_Color().blue(),
                    cv.viz_Color().gold(),
                    cv.viz_Color().raspberry(),
                    cv.viz_Color().bluberry(),
                    cv.viz_Color().lime()]
        palette_size = len(palette)
        polyline = np.zeros(shape=(1, 32, 3), dtype=np.float32)
        colors = np.zeros(shape=(1, 32, 3), dtype=np.uint8)
        for i in range(polyline.shape[1]):
            polyline[0,i,0] = i / 16.0
            polyline[0,i,1] = np.cos(i * np.pi/6)
            polyline[0,i,2] = np.sin(i * np.pi/6)
            colors[0,i,0] = palette[i % palette_size].get_blue()
            colors[0,i,1] = palette[i % palette_size].get_green()
            colors[0,i,2] = palette[i % palette_size].get_red()

        viz = cv.viz_Viz3d("show_polyline")
        viz.showWidget("polyline", cv.viz_WPolyLine(polyline, colors))
        viz.showWidget("coosys", cv.viz_WCoordinateSystem())
        viz.showWidget("text2d", cv.viz_WText("Polyline", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)

    def test_viz_show_sampled_normals(self):

        mesh  = cv.viz.readMesh(self.find_file("viz/dragon.ply"))
        mesh.normals = cv.viz.computeNormals(mesh)
        pose = cv.viz_Affine3d().rotate((0, 0.8, 0))
        viz = cv.viz_Viz3d("show_sampled_normals")
        viz.showWidget("mesh", cv.viz_WMesh(mesh), pose)
        viz.showWidget("normals", cv.viz_WCloudNormals(mesh.cloud, mesh.normals, 30, 0.1, cv.viz_Color().green()), pose)
        viz.setRenderingProperty("normals", cv.viz.LINE_WIDTH, 2.0)
        viz.showWidget("text2d", cv.viz_WText("Cloud or mesh normals", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True);


    def test_viz_show_cloud_shaded_by_normals(self):
        mesh  = cv.viz.readMesh(self.find_file("viz/dragon.ply"))
        mesh.normals = cv.viz.computeNormals(mesh)
        pose = cv.viz_Affine3d().rotate((0, 0.8, 0))

        cloud = cv.viz_WCloud(mesh.cloud, cv.viz_Color().white(), mesh.normals)
        cloud.setRenderingProperty(cv.viz.SHADING, cv.viz.SHADING_GOURAUD)
        viz = cv.viz_Viz3d("show_cloud_shaded_by_normals")

        viz.showWidget("cloud", cloud, pose)
        viz.showWidget("text2d", cv.viz_WText("Cloud shaded by normals", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)

    def test_viz_show_image_method(self):
        lena = cv.imread(self.find_file("viz/lena.png"))
        lena_gray = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)
        viz = cv.viz_Viz3d("show_image_method")
        viz.showImage(lena)
        viz.spinOnce(1500, True)
        viz.showImage(lena, (lena.shape[1], lena.shape[0]))
        viz.spinOnce(1500, True)

        #cv.viz.imshow("show_image_method", lena_gray).spinOnce(500, True) BUG

    def test_viz_show_follower(self):
        viz = cv.viz_Viz3d("show_follower")

        viz.showWidget("coos", cv.viz_WCoordinateSystem())
        viz.showWidget("cube", cv.viz_WCube())
        text_3d = cv.viz_WText3D("Simple 3D follower", (-0.5, -0.5, 0.5), 0.125, True,  cv.viz_Color().green())
        viz.showWidget("t3d_2", text_3d)
        viz.showWidget("text2d", cv.viz_WText("Follower: text always facing camera", (20, 20), 20, cv.viz_Color().green()))
        viz.setBackgroundMeshLab()
        viz.spinOnce(500, True)
        text_3d.setText("Updated follower 3D")
        viz.spinOnce(500, True)

    def test_viz_show_trajectory_reposition(self):
        mat, path = generate_test_trajectory()
        viz = cv.viz_Viz3d("show_trajectory_reposition_to_origin")
        viz.showWidget("coos", cv.viz_WCoordinateSystem())
        viz.showWidget("sub3", cv.viz_WTrajectory(mat[0: len(path) // 3,:,:], cv.viz.PyWTrajectory_BOTH, 0.2, cv.viz_Color().brown()), path[0].inv())
        viz.showWidget("text2d", cv.viz_WText("Trajectory resposition to origin", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)

    def test_viz_show_trajectories(self):
        mat, path = generate_test_trajectory()
        size =len(path)

        sub0 = np.copy(mat[0: size//10+1,::])
        sub1 = np.copy(mat[size//10: size//5+1,::])
        sub2 = np.copy(mat[size//5: 11*size//12,::])
        sub3 = np.copy(mat[11 * size // 12 :  size,::])
        sub4 = np.copy(mat[3 * size//4: 33*size//40,::])
        sub5 = np.copy(mat[11*size//12: size,::])
        K = np.array([[1024.0, 0.0, 320.0], [0.0, 1024.0, 240.0], [0.0, 0.0, 1.0]],dtype=np.float64)

        viz = cv.viz_Viz3d("show_trajectories")
        viz.showWidget("coos", cv.viz_WCoordinateSystem())
        viz.showWidget("sub0", cv.viz_WTrajectorySpheres(sub0, 0.25, 0.07))
        viz.showWidget("sub1", cv.viz_WTrajectory(sub1, cv.viz.PyWTrajectory_PATH, 0.2, cv.viz_Color().brown()))
        viz.showWidget("sub2", cv.viz_WTrajectory(sub2, cv.viz.PyWTrajectory_FRAMES, 0.2))
        viz.showWidget("sub3", cv.viz_WTrajectory(sub3, cv.viz.PyWTrajectory_BOTH, 0.2, cv.viz_Color().green()))
        viz.showWidget("sub4", cv.viz_WTrajectoryFrustums(sub4, K, 0.3, cv.viz_Color().yellow()))
        viz.showWidget("sub5", cv.viz_WTrajectoryFrustums(sub5, (0.78, 0.78), 0.15, cv.viz_Color().magenta())) #BUG
        viz.showWidget("text2d", cv.viz_WText("Different kinds of supported trajectories", (20, 20), 20, cv.viz_Color().green()))

        i = 0
        for num in range(50):
            i = i - 1
            a = i % 360
            pose = (np.sin(a * np.pi/180)* 7.5, 0.7, np.cos(a * np.pi/180)* 7.5)
            viz.setViewerPose(cv.viz.makeCameraPose(pose , (0.0, 0.5, 0.0), (0.0, 0.1, 0.0)));
            viz.spinOnce(100, True)
        viz.resetCamera()
        viz.spinOnce(500, True)

    def test_viz_show_camera_positions(self):
        K = np.array([[1024.0, 0.0, 320.0], [0.0, 1024.0, 240.0], [0.0, 0.0, 1.0]],dtype=np.float64)
        lena = cv.imread(self.find_file("viz/lena.png"))
        lena_gray = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

        poses = []
        for i in range(2):
            pose = (5 * np.sin(3.14 + 2.7 + i*60 * np.pi/180), 2 - i*1.5, 5 * np.cos(3.14 + 2.7 + i*60 * np.pi/180))
            poses.append(cv.viz.makeCameraPose(pose, (0.0, 0.0, 0.0), (0.0, -0.1, 0.0)))
        viz = cv.viz_Viz3d("show_camera_positions")
        viz.showWidget("sphe", cv.viz_WSphere((0,0,0), 1.0, 10, cv.viz_Color().orange_red()))
        viz.showWidget("coos", cv.viz_WCoordinateSystem(1.5))
        viz.showWidget("pos1", cv.viz_WCameraPosition(0.75), poses[0])
        viz.showWidget("pos2", cv.viz_WCameraPosition((0.78, 0.78), lena, 2.2, cv.viz_Color().green()), poses[0])
        viz.showWidget("pos3", cv.viz_WCameraPosition(0.75), poses[0])
        viz.showWidget("pos4", cv.viz_WCameraPosition(K, lena_gray, 3, cv.viz_Color().indigo()), poses[1])
        viz.showWidget("text2d", cv.viz_WText("Camera positions with images", (20, 20), 20, cv.viz_Color().green()))
        viz.spinOnce(500, True)
"""
TEST(Viz, show_widget_merger)
{
    WWidgetMerger merger;
    merger.addWidget(WCube(Vec3d::all(0.0), Vec3d::all(1.0), true, Color::gold()));

    RNG& rng = theRNG();
    for(int i = 0; i < 77; ++i)
    {
        Vec3b c;
        rng.fill(c, RNG::NORMAL, Scalar::all(128), Scalar::all(48), true);
        merger.addWidget(WSphere(Vec3d(c)*(1.0/255.0), 7.0/255.0, 10, Color(c[2], c[1], c[0])));
    }
    merger.finalize();

    Viz3d viz("show_mesh_random_color");
    viz.showWidget("coo", WCoordinateSystem());
    viz.showWidget("merger", merger);
    viz.showWidget("text2d", WText("Widget merger", Point(20, 20), 20, Color::green()));
    viz.spinOnce(500, true);
}



"""
if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
