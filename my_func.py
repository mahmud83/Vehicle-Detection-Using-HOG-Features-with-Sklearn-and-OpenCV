# Utility Functions I created or modified from pre-existing functions

import numpy as np
import time
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from features import *

# This is simply the training block used in the Notebook, wrapped in a function.
# This allows me to loop and perform tests on all the color spaces
def get_best_color_space(cars, notcars):
    t = time.time()
    
    # Testing with 500 images randomly generated
    n = 500
    random_indices = np.random.randint(0,len(cars),n)
    test_cars = np.array(cars)[random_indices]
    test_notcars = np.array(notcars)[random_indices]
    
    # colorspace list and empty accuracy record list
    accuracy_record = []
    color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
    
    # Create empty Pandas dataframe
    data = np.zeros((10,6))
    ac_DF = pd.DataFrame(data, columns = color_spaces)
    
    # Run 10 test
    for i in range(0,10):
        # Run tests across all color spaces 
        for j in color_spaces:
            # Parameters
            color_space = j
            orient = 9
            pix_per_cell = 8
            cell_per_block = 2
            hog_channel = 'ALL'
            spatial_size = (32,32)
            hist_bins = 32
            spatial_feat = True
            hist_feat = True
            hog_feat = True
            
            # Feature extraction  
            car_features = extract_features(test_cars, color_space=color_space, spatial_size= spatial_size,
                                        hist_bins= hist_bins, orient= orient, 
                                        pix_per_cell= pix_per_cell, cell_per_block= cell_per_block, hog_channel= hog_channel,
                                        spatial_feat= spatial_feat, hist_feat= hist_feat, hog_feat= hog_feat)
            notcar_features = extract_features(test_notcars, color_space=color_space, spatial_size= spatial_size,
                                        hist_bins= hist_bins, orient= orient, 
                                        pix_per_cell= pix_per_cell, cell_per_block= cell_per_block, hog_channel= hog_channel,
                                        spatial_feat= spatial_feat, hist_feat= hist_feat, hog_feat= hog_feat)
            
            # Scalar to normalize
            X = np.vstack((car_features,notcar_features)).astype(np.float64)
            X_scalar =  StandardScaler().fit(X)
            scaled_X = X_scalar.transform(X)
            
            # Create labels of 0 and 1 based on image type
            y =  np.hstack((np.ones(len(car_features)),np.zeros(len(notcar_features))))
            
            # Split test and train sets
            X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size =0.1)
            
            # Train with SVC
            svc = LinearSVC()
            t= time.time()
            svc.fit(X_train,y_train)
            
            # Store results in a Data Frame
            ac_DF[j][i] = round(svc.score(X_test, y_test),4)
        print('Done with Test : ', i)
    print('All tests done.') 
    return ac_DF