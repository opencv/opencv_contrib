import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import util

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.forward(data)
    w        = data['w']
    h        = data['h']
    visuals  = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    aspect_ratio = opt.aspect_ratio
    for label, im in visuals.items():
            image_name = '%s.png' % (name)
            save_path = os.path.join(image_dir, image_name)
            if aspect_ratio >= 1.0:
                 im = np.array(Image.fromarray(im).resize((h, int(w * aspect_ratio))))
            if aspect_ratio < 1.0:
                 im = np.array(Image.fromarray(im).resize((int(h/aspect_ratio),w)))
            utils.save_image(im, save_path)
