from scipy import misc
import re
import sys, os
import numpy as np

def readImage():
    pre_path = 'dataSets/part/image/'
    pattern = '(\d+).png$'
    files = sorted(os.listdir(pre_path), key = lambda x : int(re.search(pattern, x).group(1)))
    print(len(files))
    exit()
    image_list = []
    for filename in files:
        image_path = pre_path + filename
        image = misc.imread(image_path)
        image_list.append(image)
        print(type((500,333,3)))
        if image.shape != (500,333,3):
            print(filename)
            print(image.shape)
            break
    return np.array(image_list)

if __name__ == "__main__":
    image_list = readImage()
    print(image_list[1].shape)
