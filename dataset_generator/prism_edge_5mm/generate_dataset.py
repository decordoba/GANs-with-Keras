from PIL import Image
import numpy as np
import os


dirpath = "."
i = 0
for f in os.listdir(dirpath):
    # Save only tiff images
    if not f.endswith(".tif"):
        continue
    # Open image and save it in numpy array
    img = Image.open(f)
    img_list = np.array(img.getdata()).reshape(img.size[0], img.size[1])
    i += 1
    print("Image {} ({}) dimensions: {}".format(i, f, img_list.shape))
    # Save all lumpy_images
    try:
        dataset = np.concatenate((dataset, [img_list]))
    except NameError:
        dataset = np.array([img_list])

print("Created dataset with dimensions: {}".format(dataset.shape))
print("Values in dataset are in range [ {} - {} ]".format(np.min(dataset), np.max(dataset)))
print("Unique values:", np.unique(dataset))

dataset_name = "prism_edge_5mm1"
print("Saving data in '{}.npy'".format(dataset_name))
np.save(dataset_name, dataset)
