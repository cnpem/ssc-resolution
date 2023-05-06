import numpy as np

def make_image_squared(img, start, size):
    return img[start:start+size,start:start+size]

def load_data_for_FC(path='',start=0,size=10,use_phantom=True):
    if use_phantom:
        from skimage import data
        caller = getattr(data, 'camera')
        img = caller()
    else:
        datatype = path.rsplit('.')[1] 
        if datatype == 'npy':
            img = np.load(path)
        elif datatype == 'tiff':
            from PIL import Image
            img = Image.open('a_image.tif')
        else:
            sys.exit("Please select .npy or .tiff file")

    if img.shape[0] != img.shape[1]:
        print(f"Image not squared. Initial shape: {img.shape}")
        img = make_image_squared(img,start,size)
        print(f"Final shape: {img.shape}")        
    return img 
