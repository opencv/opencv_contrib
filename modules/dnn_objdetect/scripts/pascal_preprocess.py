from skimage import io, transform
from multiprocessing.dummy import Pool as ThreadPool

def rescale(root_new, root_old, img_path, ann_path, out_shape):
  try:
    img = io.imread(root_old+"/"+img_path)
  except Exception as E:
    print E
  h, w, _ = img.shape
  f_h, f_w = float(out_shape)/h, float(out_shape)/w
  trans_img = transform.rescale(img, (f_h, f_w))
  num_objs = 0
  with open(root_old+"/"+ann_path, 'r') as f:
    ann = f.readline()
    ann = ann.rstrip()
    ann = ann.split(' ')
    ann = [float(i) for i in ann]
    num_objs = len(ann) / 5
    for idx in xrange(num_objs):
      ann[idx * 5 + 0] = int(f_w * ann[idx * 5 + 0])
      ann[idx * 5 + 1] = int(f_h * ann[idx * 5 + 1])
      ann[idx * 5 + 2] = int(f_w * ann[idx * 5 + 2])
      ann[idx * 5 + 3] = int(f_h * ann[idx * 5 + 3])
    # Write the new annotations to file
    with open(root_new+"/"+ann_path, 'w') as f_new:
      for val in ann:
        f_new.write(str(val)+' ')
  # Save the new image
  io.imwrite(root_new+"/"+img_path, trans_img)

def preprocess():
  source = '/users2/Datasets/PASCAL_VOC/VOCdevkit/VOC2012_Resize/source.txt'
  root_old = '/users2/Datasets/PASCAL_VOC/VOCdevkit/VOC2012'
  root_new = '/users2/Datasets/PASCAL_VOC/VOCdevkit/VOC2012_Resize'
  out_shape = 416
  with open(source, 'r') as src:
    lines = src.readlines()
    print 'Processing {} images and annotations'.format(len(lines))
    for line in lines:
      line = line.rstrip()
      line = line.split(' ')
      img_path = line[0]
      ann_path = line[1]
      rescale(root_new, root_old, img_path, ann_path, out_shape)

if __name__ == '__main__':
  preprocess()
