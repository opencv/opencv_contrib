import pyceres
import pycolmap
import numpy as np
from hloc.utils import viz_3d

from pycolmap import cost_functions

def create_reconstruction(num_points=50, num_images=2, seed=3, noise=0):
    state = np.random.RandomState(seed)
    rec = pycolmap.Reconstruction()
    p3d = state.uniform(-1, 1, (num_points, 3)) + np.array([0, 0, 3])
    for p in p3d:
        rec.add_point3D(p, pycolmap.Track(), np.zeros(3))
    w, h = 640, 480
    cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=w,
        height=h,
        params=np.array([max(w, h) * 1.2, w / 2, h / 2]),
        camera_id=0,
    )
    rec.add_camera(cam)
    for i in range(num_images):
        im = pycolmap.Image(
            id=i,
            name=str(i),
            camera_id=cam.camera_id,
            cam_from_world=pycolmap.Rigid3d(
                pycolmap.Rotation3d(), state.uniform(-1, 1, 3)
            ),
        )
        im.registered = True
        p2d = cam.img_from_cam(
            im.cam_from_world * [p.xyz for p in rec.points3D.values()]
        )
        p2d_obs = np.array(p2d) + state.randn(len(p2d), 2) * noise
        im.points2D = pycolmap.ListPoint2D(
            [pycolmap.Point2D(p, id_) for p, id_ in zip(p2d_obs, rec.points3D)]
        )
        rec.add_image(im)
    return rec


rec_gt = create_reconstruction()


fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(
    fig, rec_gt, min_track_length=0, color="rgb(255,0,0)", points_rgb=False
)
fig.show()


# def define_problem(rec): #ORIGINAL
#     prob = pyceres.Problem()
#     loss = pyceres.TrivialLoss()
#     for im in rec.images.values():
#         cam = rec.cameras[im.camera_id]
#         for p in im.points2D:
#             print(type(pycolmap.cost_functions.ReprojErrorCost(
#                 cam.model, im.cam_from_world, p.xy
#             )))
#             cost = pycolmap.cost_functions.ReprojErrorCost(
#                 cam.model, im.cam_from_world, p.xy
#             )
#             prob.add_residual_block(
#                 cost, loss, [rec.points3D[p.point3D_id].xyz, cam.params]
#             )
#     for cam in rec.cameras.values():
#         prob.set_parameter_block_constant(cam.params)
#     return prob

def define_problem(rec): #ORIGINAL
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        for p in im.points2D:
            # print((p.xy.shape), type(p.xy), p.point3D_id)
            pt2 = p.xy.reshape((2,1))
            cost = pycolmap.cost_functions.ReprojErrorCost(
                cam.model, im.cam_from_world, pt2
            )
            # cost = pycolmap.cost_functions.ReprojErrorCost(
            #     cam.model, pt2, im.cam_from_world
            # )
            prob.add_residual_block(
                cost, loss, [rec.points3D[p.point3D_id].xyz, cam.params]
            )
    for cam in rec.cameras.values():
        prob.set_parameter_block_constant(cam.params)
    return prob

def solve(prob):
    print(
        prob.num_parameter_blocks(),
        prob.num_parameters(),
        prob.num_residual_blocks(),
        prob.num_residuals(),
    )
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print(summary.BriefReport())


rec = create_reconstruction()
problem = define_problem(rec)
solve(problem)