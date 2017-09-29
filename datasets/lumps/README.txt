Steps to generate lumps data:

1. Edit dataGenerator.m and set FIRST_SAMPLE_NUM and LAST_SAMPLE_NUM to select the number
   and range of samples.

2. Run dataGenerator.m (it will create tow folders: dataset and lumps. dataset has all the
   image data, lumps saves some of the images generation data in case we want to use it later
   in our networks (it holds the locations of each lump)).

3. Run 'python3 load_data.py', it will get all the information in datasets and save it in the
   lumps.npy file, which has the right numpy format to be used in Keras.