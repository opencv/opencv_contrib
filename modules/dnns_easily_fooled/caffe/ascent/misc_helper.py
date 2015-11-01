#! /usr/bin/env python

from pylab import *



def figsize(width,height):
    rcParams['figure.figsize'] = (width,height)



def norm01(arr):
    arr = arr.copy()
    arr -= arr.min()
    arr /= arr.max()
    return arr



def norm01c(arr, center):
    '''Maps the center value to .5'''
    arr = arr.copy()
    arr -= center
    arr /= max(2 * arr.max(), -2 * arr.min())
    arr += .5
    assert arr.min() >= 0
    assert arr.max() <= 1
    return arr



def showimage(im, c01=False, bgr=False):
    if c01:
        # switch order from c,0,1 -> 0,1,c
        im = im.transpose((1,2,0))
    if im.ndim == 3 and bgr:
        # Change from BGR -> RGB
        im = im[:, :, ::-1]
    plt.imshow(im)
    #axis('tight')

def showimagesc(im, c01=False, bgr=False):
    showimage(norm01(im), c01=c01, bgr=bgr)



def saveimage(filename, im):
    matplotlib.image.imsave(filename, im)

def saveimagesc(filename, im):
    saveimage(filename, norm01(im))

def saveimagescc(filename, im, center):
    saveimage(filename, norm01c(im, center))



def tile_images(data, padsize=1, padval=0, c01=False, width=None):
    '''take an array of shape (n, height, width) or (n, height, width, channels)
    and visualize each (height, width) thing in a grid. If width = None, produce
    a square image of size approx. sqrt(n) by sqrt(n), else calculate height.'''
    data = data.copy()
    if c01:
        # Convert c01 -> 01c
        data = data.transpose(0, 2, 3, 1)
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    if width == None:
        width = int(np.ceil(np.sqrt(data.shape[0])))
        height = width
    else:
        assert isinstance(width, int)
        height = int(np.ceil(float(data.shape[0]) / width))
    padding = ((0, width*height - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((height, width) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((height * data.shape[1], width * data.shape[3]) + data.shape[4:])
    data = data[0:-padsize, 0:-padsize]  # remove excess padding
    
    return data



def vis_square(data, padsize=1, padval=0, c01=False):
    data = tile_images(data, padsize, padval, c01)
    showimage(data, c01=False)



def shownet(net):
    '''Print some stats about a net and its activations'''
    
    print '%-41s%-31s%s' % ('', 'acts', 'act diffs')
    print '%-45s%-31s%s' % ('', 'params', 'param diffs')
    for k, v in net.blobs.items():
        if k in net.params:
            params = net.params[k]
            for pp, blob in enumerate(params):
                if pp == 0:
                    print '  ', 'P: %-5s'%k,
                else:
                    print ' ' * 11,
                print '%-32s' % repr(blob.data.shape),
                print '%-30s' % ('(%g, %g)' % (blob.data.min(), blob.data.max())),
                print '(%g, %g)' % (blob.diff.min(), blob.diff.max())
        print '%-5s'%k, '%-34s' % repr(v.data.shape),
        print '%-30s' % ('(%g, %g)' % (v.data.min(), v.data.max())),
        print '(%g, %g)' % (v.diff.min(), v.diff.max())
