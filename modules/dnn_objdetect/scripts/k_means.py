import argparse
import sys
import os
import time
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def k_means(K, data, max_iter, n_jobs, image_file):
  X = np.array(data)
  np.random.shuffle(X)
  begin = time.time()
  print 'Running kmeans'
  kmeans = KMeans(n_clusters=K, max_iter=max_iter, n_jobs=n_jobs, verbose=1).fit(X)
  print 'K-Means took {} seconds to complete'.format(time.time()-begin)
  step_size = 0.2
  xmin, xmax = X[:, 0].min()-1, X[:, 0].max()+1
  ymin, ymax = X[:, 1].min()-1, X[:, 1].max()+1
  xx, yy = np.meshgrid(np.arange(xmin, xmax, step_size), np.arange(ymin, ymax, step_size))
  preds = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
  preds = preds.reshape(xx.shape)

  plt.figure()
  plt.clf()
  plt.imshow(preds, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
  plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
  centroids = kmeans.cluster_centers_
  plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=5, color='r', zorder=10)
  plt.title("Anchor shapes generated using K-Means")
  plt.xlim(xmin, xmax)
  plt.ylim(ymin, ymax)
  print 'Mean centroids are:'
  for i, center in enumerate(centroids):
    print '{}: {}, {}'.format(i, center[0], center[1])
  # plt.xticks(())
  # plt.yticks(())
  plt.show()

def pre_process(directory, data_list):
  if not os.path.exists(directory):
    print "Path {} doesn't exist".format(directory)
    return
  files = os.listdir(directory)
  print 'Loading data...'
  for i, f in enumerate(files):
    # Progress bar
    sys.stdout.write('\r')
    percentage = (i+1.0) / len(files)
    progress = int(percentage * 30)
    bar = [progress*'=', ' '*(29-progress), percentage*100]
    sys.stdout.write('[{}>{}]  {:.0f}%'.format(*bar))
    sys.stdout.flush()

    with open(directory+"/"+f, 'r') as ann:
      l = ann.readline()
      l = l.rstrip()
      l = l.split(' ')
      l = [float(i) for i in l]
      if len(l) % 5 != 0:
        sys.stderr.write('File {} contains incorrect number of annotations'.format(f))
        return
      num_objs = len(l) / 5
      for obj in range(num_objs):
        xmin = l[obj * 5 + 0]
        ymin = l[obj * 5 + 1]
        xmax = l[obj * 5 + 2]
        ymax = l[obj * 5 + 3]
        w = xmax - xmin
        h = ymax - ymin
        data_list.append([w, h])
        if w > 1000 or h > 1000:
          sys.stdout.write("[{}, {}]".format(w, h))
  sys.stdout.write('\nProcessed {} files containing {} objects'.format(len(files), len(data_list)))
  return data_list

def main():
  parser = argparse.ArgumentParser("Parse hyperparameters")
  parser.add_argument("clusters", help="Number of clusters", type=int)
  parser.add_argument("dir", help="Directory containing annotations")
  parser.add_argument("image_file", help="File to generate the final cluster of image")
  parser.add_argument('-jobs', help="Number of jobs for parallel computation", default=1)
  parser.add_argument('-iter', help="Max Iterations to run algorithm for", default=1000)

  p = parser.parse_args(sys.argv[1:])
  K = p.clusters
  directory = p.dir
  data_list = []
  pre_process(directory, data_list  )
  sys.stdout.write('\nDone collecting data\n')
  k_means(K, data_list, int(p.iter), int(p.jobs), p.image_file)
  print 'Done !'

if __name__=='__main__':
  try:
    main()
  except Exception as E:
    print E
