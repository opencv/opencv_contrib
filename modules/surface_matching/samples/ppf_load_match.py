import cv2 as cv

N = 2
modelname = "parasaurolophus_6700"
scenename = "rs1_normals"

detector = cv.ppf_match_3d_PPF3DDetector(0.025, 0.05)

print('Loading model...')
pc = cv.ppf_match_3d.loadPLYSimple("data/%s.ply" % modelname, 1)


print('Training...')
detector.trainModel(pc)

print('Loading scene...')
pcTest = cv.ppf_match_3d.loadPLYSimple("data/%s.ply" % scenename, 1)

print('Matching...')
results = detector.match(pcTest, 1.0/40.0, 0.05)

print('Performing ICP...')
icp = cv.ppf_match_3d_ICP(100)
_, results = icp.registerModelToScene(pc, pcTest, results[:N])

print("Poses: ")
for i, result in enumerate(results):
    #result.printPose()
    print("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex, result.numVotes, result.residual, result.pose))
    if i == 0:
        pct = cv.ppf_match_3d.transformPCPose(pc, result.pose)
        cv.ppf_match_3d.writePLY(pct, "%sPCTrans.ply" % modelname)
