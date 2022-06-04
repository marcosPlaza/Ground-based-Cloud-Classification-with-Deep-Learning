import imageio
import os
from os.path import exists
import traceback
from tqdm import tqdm

root = "/Users/marcosplazagonzalez/Desktop/Ground-based_CloudClassification/Datasets/DataAlfons/"
dest = "/Users/marcosplazagonzalez/Desktop/Ground-based_CloudClassification/Datasets/Splitted/"

for dirname, _, filenames in tqdm(list(os.walk(root))):
    for filename in filenames:
        if filename == '.DS_Store': continue

        path = os.path.join(dirname, filename) 
        splits = dirname.split('/')
        class_name = splits[-1]

        img = imageio.imread(path)

        height, width, n_channel = img.shape

        if height > width: # if the image is taller than it is wide save and continue
            imageio.imsave(dest+class_name+"/"+filename, img)
            continue

        # Cut the image in half => Adding margins to the image
        width_cutoff = width // 2

        s1 = img[50:-50, :width_cutoff-int(0.1*width_cutoff)]
        s2 = img[50:-50, width_cutoff+int(0.1*width_cutoff):]

        withoutext = filename.split('.')[0]

        # Save each half
        path_a = dest+class_name+"/"+withoutext+"_s1.jpg"
        path_b = dest+class_name+"/"+withoutext+"_s2.jpg"
        
        try:
            if not exists(path_a): imageio.imsave(path_a, s1)
            if not exists(path_b): imageio.imsave(path_b, s2)
                
        except Exception:
            # must convert RGBA to RGB
            #path_a = dest+class_name+"/"+withoutext+"_s1.png"
            #path_b = dest+class_name+"/"+withoutext+"_s2.png"
        
            #if not exists(path_a): imageio.imsave(path_a, s1)
            #if not exists(path_b): imageio.imsave(path_b, s2)
            traceback.print_exc()
