import os
from skimage import io

def load_map(name = 'racetrack.png'):
    image_path = os.path.join(os.getcwd(),'map/'+name)
    # img = mpimg.imread(path)

    img = io.imread(image_path, as_gray=True)/255.0
    return img