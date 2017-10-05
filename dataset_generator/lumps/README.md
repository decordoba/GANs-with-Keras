# Steps to generate lumps data

1. Edit `dataGenerator.m` and set `FIRST_SAMPLE_NUM` and `LAST_SAMPLE_NUM` to select the number
   and range of samples, as well as the parameters to generate the lumps.

2. Run `dataGenerator.m` executing `matlab -r "run('dataGenerator()')"` (it will create two folders: `dataset` and `lumps`. `dataset` is where all the
   image data is stored, and `lumps` contains some of the images generation data in case we want to use it later
   in our networks (it holds the locations of each lump)). It is possible to see the data instead of saving it
   by running `matlab -r "run('dataGenerator(false)')"`, but then nothing will be saved inside the two folders mentioned. This
   can be used for debugging purposes, to check the kind of images that will be generated before creating the dataset.

3. Run 'python3 load_data.py', it will get all the information in `dataset` and save it in the
   `lumps.npy` file, which has the right `numpy` format to be used in Keras.