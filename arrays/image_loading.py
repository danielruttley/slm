import numpy as np
from PIL import Image

def scale_image(array,scale):
    [ys,xs] = np.where(array)
    traps = [(y,x) for y,x in zip(ys,xs)]
    scaled_array = np.zeros(tuple(int(round(i*scale)) for i in array.shape),dtype=bool)
    traps = [(int(round(y*scale)),int(round(x*scale))) for (y,x) in traps]
    for trap in traps:
        # print(trap)
        scaled_array[trap] = 1
    return scaled_array

def crop_image(array):
    [ys,xs] = np.where(array)
    return array[np.min(ys):np.max(ys)+1,np.min(xs):np.max(xs)+1]

def pad_image(array,shape):
    (y_pad,x_pad) = shape
    (y_array,x_array) = array.shape
    y_needed = y_pad - y_array
    x_needed = x_pad - x_array
    y1 = int(round(y_needed/2))
    y2 = y_needed - y1
    x1 = int(round(y_needed/2))
    x2 = x_needed - x1
    return np.pad(array,((y1,y2),(x1,x2)))

def prepare_image(array,scale=1,shape=(512,512)):
    array = scale_image(array,scale)
    array = crop_image(array)
    # array = pad_image(array,shape)
    return array

def load_traps_from_image(filename,scale=1,shape=(512,512)):
    img = Image.open(filename)
    img = img.convert('1')
    array = np.asarray(img,dtype=bool)
    array = np.flipud(array)
    array = prepare_image(array,scale,shape)
    [ys,xs] = np.where(array)
    ymin = 240
    xmin = 240
    traps = [(x+xmin,y+ymin) for y,x in zip(ys,xs)]
    return traps

if __name__ == '__main__':
    scale = 4
    shape = (512,512)
    traps = load_traps_from_image('angel_simon.png',scale,shape)
    print(traps)
    print(len(traps))
    array = np.zeros(shape)
    for trap in traps:
        array[trap] = 1