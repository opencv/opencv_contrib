# Python wrapper by: Hamdi Sahloul <hamdisahloul AT hotmail.com>

import cv2;
import numpy as np;
import sys;

MATCH_X = 0;
MATCH_Y = 1;
MATCH_SIMILARITY = 2;
MATCH_CLASS_ID = 3;
MATCH_TEMPLATE_ID = 4;

TEMPLATE_FEATURES = 3;

FEATURE_X = 0;
FEATURE_Y = 1;

# Copy of cv_mouse from cv_utilities
class Mouse:
    m_event = -1;
    m_x = 0;
    m_y = 0;

    @classmethod
    def start(cls, a_img_name):
            cv2.setMouseCallback(a_img_name, cls.cv_on_mouse, 0);

    @classmethod
    def event(cls):
        l_event = cls.m_event;
        cls.m_event = -1;
        return l_event;

    @classmethod
    def x(cls):
        return cls.m_x;

    @classmethod
    def y(cls):
        return cls.m_y;

    @classmethod
    def cv_on_mouse(cls, a_event, a_x, a_y, *args, **kwargs):
        cls.m_event = a_event;
        cls.m_x = a_x;
        cls.m_y = a_y;

def help():
    print("Usage: openni_demo.py\n\n" # [templates.yml]
                 "Place your object on a planar, featureless surface. With the mouse,\n"
                 "frame it in the 'color' window and right click to learn a first template.\n"
                 "Then press 'l' to enter online learning mode, and move the camera around.\n"
                 "When the match score falls between 90-95%% the demo will add a new template.\n\n"
                 "Keys:\n"
                 "\t h   -- This help page\n"
                 "\t l   -- Toggle online learning\n"
                 "\t m   -- Toggle printing match result\n"
                 "\t t   -- Toggle printing timings\n"
                 #"\t w   -- Write learned templates to disk\n"
                 "\t [ ] -- Adjust matching threshold: '[' down,  ']' up\n"
                 "\t q   -- Quit\n\n");

# Adapted from cv_timer in cv_utilities
class Timer:
    def __init__(self, *args, **kwargs):
        self.start_ = 0;
        self.time_ = 0;

    def start(self):
        self.start_ = cv2.getTickCount();

    def stop(self):
        assert(self.start_ is not 0);
        end = cv2.getTickCount();
        self.time_ += end - self.start_;
        self.start_ = 0;

    def time(self):
        ret = self.time_ / cv2.getTickFrequency();
        time_ = 0;
        return ret;

# Functions to store detector and templates in single XML/YAML file
#def readLinemod(filename):
#    detector = cv2.linemod.Detector();
#    fs = cv2.FileStorage(filename, cv2.FileStorage.READ);
#    detector.read(fs.root());
#
#    fn = fs["classes"];
#    for i in fn:
#        detector.readClass(i);
#
#    return detector;

#def writeLinemod(detector, filename):
#    fs = cv2.FileStorage(filename, 1);
#    detector.write(fs);
#
#    ids = detector.classIds();
#    fs.write("classes" + "[");
#    for id in ids:
#        fs.write("{");
#        detector.writeClass(id, fs);
#        fs.write("}"); # current class
#    fs.write("]"); # classes

def subtractPlane(depth, chain, f):
    mask = np.zeros(depth.shape, np.uint8);
    a_masks = filterPlane(depth, [mask], chain, f);
    return a_masks[0]

def maskFromTemplate(templates, num_modalities, offset, size, dst):
    mask = templateConvexHull(templates, num_modalities, offset, size);

    OFFSET = 30;
    mask = cv2.dilate(mask, None, iterations=OFFSET);

    retval = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    if len(retval) == 3: retval = retval[1:];
    contours, hierarchy = retval;
    l_pts1 = contours[0].reshape((-1, 2));

    cv2.polylines(dst, pts=[l_pts1], isClosed=True, color=(0, 255, 0), thickness=2);

    return l_pts1, mask;

# Adapted from cv_show_angles
def displayQuantized(quantized):
    color = np.ndarray((quantized.shape[0], quantized.shape[1], 3), np.uint8);
    for r in range(quantized.rows):
        quant_r = quantized.ptr(r);
        color_r = color.ptr<cv2.Vec3b>(r);

        for c in range(quantized.cols):
            bgr = color_r[c];
            if quant_r[c] == 0: bgr[0]= 0; bgr[1]= 0; bgr[2]= 0;
            elif quant_r[c] == 1: bgr[0]= 55; bgr[1]= 55; bgr[2]= 55;
            elif quant_r[c] == 2: bgr[0]= 80; bgr[1]= 80; bgr[2]= 80;
            elif quant_r[c] == 4: bgr[0]=105; bgr[1]=105; bgr[2]=105;
            elif quant_r[c] == 8: bgr[0]=130; bgr[1]=130; bgr[2]=130;
            elif quant_r[c] == 16: bgr[0]=155; bgr[1]=155; bgr[2]=155;
            elif quant_r[c] == 32: bgr[0]=180; bgr[1]=180; bgr[2]=180;
            elif quant_r[c] == 64: bgr[0]=205; bgr[1]=205; bgr[2]=205;
            elif quant_r[c] == 128: bgr[0]=230; bgr[1]=230; bgr[2]=230;
            elif quant_r[c] == 255: bgr[0]= 0; bgr[1]= 0; bgr[2]=255;
            else: bgr[0]= 0; bgr[1]=255; bgr[2]= 0;

    return color;

# Adapted from cv_line_template.convex_hull
def templateConvexHull(templates, num_modalities, offset, size):
    points = np.ndarray((num_modalities, 2), np.int);
    for m in range(num_modalities):
        for i in range(len(templates[m][TEMPLATE_FEATURES])):
            f = templates[m][TEMPLATE_FEATURES][i];
            points[m] = f[0:2] + offset;

    hull = cv2.convexHull(points);

    dst = np.zeros(size, np.uint8);
    cv2.fillPoly(dst, [hull[0]], 255);
    return dst

def drawResponse(templates, num_modalities, dst, offset, T):
    COLORS = [ np.array([255, 0, 0]),
               np.array([0, 255, 0]),
               np.array([0, 255, 255]),
               np.array([0, 140, 255]),
               np.array([0, 0, 255]) ];

    for m in range(num_modalities):
        # NOTE: Original demo recalculated max response for each feature in the TxT
        # box around it and chose the display color based on that response. Here
        # the display color just depends on the modality.
        color = COLORS[m];

        for i in range(len(templates[m][TEMPLATE_FEATURES])):
            f = templates[m][TEMPLATE_FEATURES][i];
            pt = tuple([f[FEATURE_X] + offset[0], f[FEATURE_Y] + offset[1]]);
            cv2.circle(dst, pt, T / 2, color);

def reprojectPoints(proj, f):
    f_inv = 1.0 / f;

    real = np.ndarray(proj.shape, proj.dtype);
    for i in range(proj.shape[0]):
        Z = proj[i, 2];
        real[i, 0] = (proj[i, 0] - 320.) * (f_inv * Z);
        real[i, 1] = (proj[i, 1] - 240.) * (f_inv * Z);
        real[i, 2] = Z;
    return real

def filterPlane(ap_depth, a_masks, a_chain, f):
    l_num_cost_pts = 200;

    l_thres = 4;

    lp_mask = np.zeros(ap_depth.shape, np.uint8);

    l_chain_vector = [];

    l_chain_length = 0;
    lp_seg_length = [];

    for l_i in range(a_chain.shape[0]):
        x_diff = float(a_chain[(l_i + 1) % a_chain.shape[0], 0] - a_chain[l_i, 0]);
        y_diff = float(a_chain[(l_i + 1) % a_chain.shape[0], 1] - a_chain[l_i, 1]);
        lp_seg_length.append(np.sqrt(x_diff*x_diff + y_diff*y_diff));
        l_chain_length += lp_seg_length[l_i];
    for l_i in range(a_chain.shape[0]):
        if lp_seg_length[l_i] > 0:
            l_cur_num = np.rint(l_num_cost_pts * lp_seg_length[l_i] / l_chain_length).astype(int);
            l_cur_len = lp_seg_length[l_i] / l_cur_num;

            for l_j in range(l_cur_num):
                l_ratio = (l_cur_len * l_j / lp_seg_length[l_i]);

                l_pts = (np.rint(l_ratio * (a_chain[(l_i + 1) % a_chain.shape[0], 0] - a_chain[l_i, 0]) + a_chain[l_i, 0]).astype(int),
                         np.rint(l_ratio * (a_chain[(l_i + 1) % a_chain.shape[0], 1] - a_chain[l_i, 1]) + a_chain[l_i, 1]).astype(int));

                l_chain_vector.append(l_pts);

    lp_src_3Dpts = np.ndarray((len(l_chain_vector), 3));
    for l_i in range(lp_src_3Dpts.shape[0]):
        lp_src_3Dpts[l_i, 0] = l_chain_vector[l_i][0];
        lp_src_3Dpts[l_i, 1] = l_chain_vector[l_i][1];
        lp_src_3Dpts[l_i, 2] = ap_depth[np.rint(lp_src_3Dpts[l_i, 1]).astype(int), np.rint(lp_src_3Dpts[l_i, 0]).astype(int)];
        #lp_mask[int(lp_src_3Dpts[l_i, 1]),int(lp_src_3Dpts[l_i, 0])]=255;
    #cv2.imshow("hallo2",lp_mask);

    lp_src_3Dpts = reprojectPoints(lp_src_3Dpts, f);

    lp_pts = np.hstack((lp_src_3Dpts, np.ones((lp_src_3Dpts.shape[0], 1))));
    _, _, lp_v = np.linalg.svd(lp_pts);

    l_n = lp_v[0:lp_src_3Dpts.shape[0], 3];
    l_n /= np.linalg.norm(l_n);

    l_max_dist = np.abs(np.sum(l_n * lp_pts, axis=1)).max();
    del lp_pts;
    del lp_v;

    #print("plane: %f;%f;%f;%f maxdist: %f end" % (l_n[0], l_n[1], l_n[2], l_n[3], l_max_dist));
    l_minx = ap_depth.shape[1];
    l_miny = ap_depth.shape[0];
    l_maxx = 0;
    l_maxy = 0;

    for l_i in range(a_chain.shape[0]):
        l_minx = np.fmin(l_minx, a_chain[l_i,0]);
        l_miny = np.fmin(l_miny, a_chain[l_i, 1]);
        l_maxx = np.fmax(l_maxx, a_chain[l_i, 0]);
        l_maxy = np.fmax(l_maxy, a_chain[l_i, 1]);
    l_w = l_maxx - l_minx + 1;
    l_h = l_maxy - l_miny + 1;
    l_nn = a_chain.shape[0];

    cv2.fillPoly(lp_mask, [a_chain], (255, 255, 255));

    #cv2.imshow("hallo1",lp_mask);

    lp_dst_3Dpts = np.ndarray((l_h * l_w, 3));

    l_ind = 0;

    for l_r in range(l_h):
        for l_c in range(l_w):
            lp_dst_3Dpts[l_ind, 0] = l_c + l_minx;
            lp_dst_3Dpts[l_ind, 1] = l_r + l_miny;
            lp_dst_3Dpts[l_ind, 2] = ap_depth[l_r + l_miny, l_c + l_minx];
            l_ind += 1;
    lp_dst_3Dpts = reprojectPoints(lp_dst_3Dpts, f);

    lp_pts = np.hstack((lp_dst_3Dpts, np.ones((lp_dst_3Dpts.shape[0], 1))));

    l_ind = 0;

    for l_r in range(l_h):
        for l_c in range(l_w):
            l_dist = np.fabs(np.sum(l_n * lp_pts[l_ind]));
            l_dist *= 3; #TODO: Fix this

            l_ind += 1;

            if lp_mask[l_r + l_miny, l_c + l_minx] is not 0:
                if l_dist < np.fmax(l_thres, (l_max_dist * 2.0)):
                    for l_p in range(len(a_masks)):
                        l_col = np.rint((l_c + l_minx) / (l_p + 1.0)).astype(int);
                        l_row = np.rint((l_r + l_miny) / (l_p + 1.0)).astype(int);

                        a_masks[l_p][l_row, l_col] = 0;
                else:
                    for l_p in range(len(a_masks)):
                        l_col = np.rint((l_c + l_minx) / (l_p + 1.0)).astype(int);
                        l_row = np.rint((l_r + l_miny) / (l_p + 1.0)).astype(int);

                        a_masks[l_p][l_row, l_col] = 255;
    del lp_mask;
    return a_masks

if __name__ == "__main__":
    # Various settings and flags
    show_match_result = True;
    show_timings = False;
    learn_online = False;
    num_classes = 0;
    matching_threshold = 80;
    roi_size = np.array([200, 200]);
    learning_lower_bound = 90;
    learning_upper_bound = 95;

    # Timers
    extract_timer = Timer();
    match_timer = Timer();

    # Initialize HighGUI
    help();
    cv2.namedWindow("color");
    cv2.namedWindow("normals");
    Mouse.start("color");

    # Initialize LINEMOD data structures
    detector = cv2.linemod.Detector();
    if len(sys.argv) == 1:
    #    filename = "linemod_templates.yml";
        detector = cv2.linemod.getDefaultLINEMOD();
    #else:
    #    detector = readLinemod(sys.argv[1]);
    #
    #    ids = detector.classIds();
    #    num_classes = detector.numClasses();
    #    print("Loaded %s with %d classes and %d templates\n" %
    #                 (sys.argv[1], num_classes, detector.numTemplates()));
    #    if len(ids):
    #        print("Class ids:\n");
    #        print(ids)
    num_modalities = len(detector.getModalities());

    # Open Kinect sensor
    capture = cv2.VideoCapture(cv2.CAP_OPENNI2);
    if not capture.isOpened():
            capture.open(cv2.CAP_OPENNI);
    if not capture.isOpened():
        print("Could not open OpenNI-capable sensor\n");
        sys.exit(-1);
    capture.set(cv2.CAP_PROP_OPENNI_REGISTRATION, 1);
    focal_length = capture.get(cv2.CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH);
    #print("Focal length = %f\n" % focal_length);

    # Main loop
    while True:
        # Capture next color/depth pair
        capture.grab();
        _, depth = capture.retrieve(0, cv2.CAP_OPENNI_DEPTH_MAP);
        _, color = capture.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE);

        sources = [];
        sources.append(color);
        sources.append(depth);
        display = color.copy();

        if not learn_online:
            mouse = np.array([Mouse.x(), Mouse.y()]);
            event = Mouse.event();

            # Compute ROI centered on current mouse location
            roi_offset = 0.5 * roi_size;
            pt1 = tuple((mouse - roi_offset).astype(np.int)); # top left
            pt2 = tuple((mouse + roi_offset).astype(np.int)); # bottom right

            if event == cv2.EVENT_RBUTTONDOWN:
                # Compute object mask by subtracting the plane within the ROI
                chain = np.array([pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])], np.int);
                mask = subtractPlane(depth, chain, focal_length);

                cv2.imshow("mask", mask);

                # Extract template
                class_id = "class" + str(num_classes);
                extract_timer.start();
                template_id, bb = detector.addTemplate(sources, class_id, mask);
                extract_timer.stop();
                if template_id is not -1:
                    print("*** Added template (id %d) for new object class %d***\n" %
                                 (template_id, num_classes));
                    #print("Extracted at (%d, %d) size %dx%d\n" % (bb.x, bb.y, bb.width, bb.height));

                num_classes += 1;

            # Draw ROI for display
            cv2.rectangle(display, pt1, pt2, (0,0,0), 3);
            cv2.rectangle(display, pt1, pt2, (0, 255, 255), 1);

        # Perform matching
        class_ids = [];
        match_timer.start();
        matches, quantized_images = detector.match(sources, float(matching_threshold), class_ids);
        match_timer.stop();

        classes_visited = 0;
        visited = [];

        for i in range(len(matches)):
            if classes_visited >= num_classes: break;
            m = matches[i];

            if m[MATCH_CLASS_ID] not in visited:
                visited.append(m[MATCH_CLASS_ID])
                classes_visited += 1;

                if show_match_result:
                    print("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n" %
                                 (m[MATCH_SIMILARITY], m[MATCH_X], m[MATCH_Y], m[MATCH_CLASS_ID], m[MATCH_TEMPLATE_ID]));

                # Draw matching template
                templates = detector.getTemplates(m[MATCH_CLASS_ID], m[MATCH_TEMPLATE_ID]);
                drawResponse(templates, num_modalities, display, ([m[MATCH_X], m[MATCH_Y]]), detector.getT(0));

                if learn_online == True:
                    #/ @todo Online learning possibly broken by new gradient feature extraction,
                    #/ which assumes an accurate object outline.

                    # Compute masks based on convex hull of matched template
                    chain, _ = maskFromTemplate(templates, num_modalities,
                                             np.array([m[MATCH_X], m[MATCH_Y]]), color.shape[0:2], display);
                    depth_mask = subtractPlane(depth, chain, focal_length);

                    cv2.imshow("mask", depth_mask);

                    # If pretty sure (but not TOO sure), add new template
                    if learning_lower_bound < m[MATCH_SIMILARITY] and m[MATCH_SIMILARITY] < learning_upper_bound:
                        extract_timer.start();
                        template_id, _ = detector.addTemplate(sources, m[MATCH_CLASS_ID], depth_mask);
                        extract_timer.stop();
                        if template_id is not -1:
                            print("*** Added template (id %d) for existing object class %s***\n" %
                                         (template_id, m[MATCH_CLASS_ID]));

        if show_match_result and len(matches) == 0:
            print("No matches found...\n");
        if show_timings:
            print("Training: %.2fs\n" % extract_timer.time());
            print("Matching: %.2fs\n" % match_timer.time());
        if show_match_result or show_timings:
            print("------------------------------------------------------------\n");

        cv2.imshow("color", display);
        cv2.imshow("normals", quantized_images[1]);

        #fs = cv2.FileStorage();
        key = chr(cv2.waitKey(10));
        if key == 'q':
            break;
        elif key == 'h':
            help();
        elif key == 'm':
            # toggle printing match result
            show_match_result = not show_match_result;
            print("Show match result %s\n" % ("ON" if show_match_result else "OFF"));
        elif key == 't':
            # toggle printing timings
            show_timings = not show_timings;
            print("Show timings %s\n" % ("ON" if show_timings else "OFF"));
        elif key == 'l':
            # toggle online learning
            learn_online = not learn_online;
            print("Online learning %s\n" % ("ON" if learn_online else "OFF"));
        elif key == '[':
            # decrement threshold
            matching_threshold = np.fmax(matching_threshold - 1, -100);
            print("New threshold: %d\n" % matching_threshold);
        elif key == ']':
            # increment threshold
            matching_threshold = np.fmin(matching_threshold + 1, +100);
            print("New threshold: %d\n" % matching_threshold);
        #elif key == 'w':
        #    # write model to disk
        #    writeLinemod(detector, filename);
        #    print("Wrote detector and templates to %s\n" % filename);
