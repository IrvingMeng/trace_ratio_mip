# ----------------------------------------------------------------------
# Code for test the trace ratio problem
# Test on the ORL dataset
# This dataset contains a set of face images taken between April 1992 and April 1994 at AT&T Laboratories Cambridge. The sklearn.datasets.fetch_olivetti_faces function is the data fetching / caching function that downloads the data archive from AT&T.
# There are ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement).
# The image is quantized to 256 grey levels and stored as unsigned 8-bit integers; the loader will convert these to floating point values on the interval [0, 1], which are easier to work with for many algorithms.
# The target for this database is an integer from 0 to 39 indicating the identity of the person pictured; however, with only 10 examples per class, this relatively small dataset is more interesting from an unsupervised or semi-supervised perspective.
# The original dataset consisted of 92 x 112, while the version available here consists of 64x64 images.
# When using these images, please give credit to AT&T Laboratories Cambridge.
# ----------------------------------------------------------------------
# This code will include these parts: 
# [DONE] 1. Pre-process the data 
# [DONE] 2. Save the dig_B, dig_E 
# [DONE] 3. Save the train test
# [DONE] 4. Give a function for split the dataset randomly, return the 
#           train_x, train_y, test_x, test_y, dig_B, dig_E
# [DONE] 5. Add a function for adding class seperated constraints
# ----------------------------------------------------------------------

import numpy as np
from random import shuffle
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import seaborn

def data_orl_random_split():
    """
    #  load the data
    #  build train and test 1:1 randomly
    #  return train_x,y, test_x,y, dig_B, dig_E
    """
    # Load the dataset
    # print 'load the data...'
    orl_data = fetch_olivetti_faces()
    targets = orl_data.target # 400, 40 people,  10 images each
    data = orl_data.images.reshape((len(orl_data.images), -1)).T # (4096,400), 4096 = 64*64

    unique_y = np.unique(targets)
    K = len(unique_y)
    d, n = data.shape


    # print '    Split the data...'

    choose_list = []
    for k in range(K):
        choose_list.append(np.where(targets == k))
    # split the train/test data
    n_k = []
    mu_k = []
    Sigma_k = []
    for k in range(K):    
        tmp_list = choose_list[k][0]
        shuffle(tmp_list)
        tmp_n = len(tmp_list)
        tmp_train_x = data[:,tmp_list[0:int(tmp_n/2)]]
        tmp_train_y = targets[tmp_list[0:int(tmp_n/2)]]
        tmp_test_x = data[:,tmp_list[int(tmp_n/2)+1:-1]]
        tmp_test_y = targets[tmp_list[int(tmp_n/2)+1:-1]]
  
        n_k.append(tmp_train_x.shape[1])
        mu_k.append(np.mean(tmp_train_x,axis=1))
        Sigma_k.append(np.cov(tmp_train_x))
    
        if k == 0:
            train_x = tmp_train_x
            train_y = tmp_train_y
            test_x = tmp_test_x
            test_y = tmp_test_y
        else:
            train_x = np.concatenate([train_x, tmp_train_x],axis=1)
            train_y = np.concatenate([train_y, tmp_train_y])
            test_x = np.concatenate([test_x, tmp_test_x],axis=1)
            test_y = np.concatenate([test_y, tmp_test_y])


    # calculate the two matrix
    mu = np.zeros([d])
    for k in range(K):
        mu = mu+n_k[k]*mu_k[k] 
    mu = mu/sum(n_k)

    B = np.zeros([d,d])
    E = np.zeros([d,d])
    for k in range(K):
        E = E + n_k[k]*Sigma_k[k]
        B = B + n_k[k]*np.outer(mu_k[k]-mu,mu_k[k]-mu)

    dig_E = []
    dig_B = []
    B_E_ratio = []
    for i in range(d):
        dig_B.append(B[i,i])
        dig_E.append(E[i,i])
    dig_B = np.array(dig_B)
    dig_E = np.array(dig_E)
    return train_x, train_y, test_x, test_y, dig_B, dig_E

def data_orl_add_constraints(threshold=0.95):
    """
    create the matrix for the orl dataset and constraints
    """
    # print 'load the data...'
    orl_data = fetch_olivetti_faces()
    targets = orl_data.target # 400, 40 people,  10 images each
    data = orl_data.images.reshape((len(orl_data.images), -1)).T # (4096,400), 4096 = 64*64
    c = np.corrcoef(data)
    
    # # show the plot
    # plt.imshow(c,interpolation='nearest')
    # plt.colorbar()
    # plt.show()

    d = c.shape[0]
    # here if feature i and j are highly correlated, add p_i + p_j \leq 1
    # So cons_var.append([i,j]), cons_coef.append([1,1]), cons_rhs.append(1)
    cons_var = []
    cons_coef = []
    cons_rhs = []
    ignore_list = []
    for i in range(d):
        for j in range(i):
            if (c[i,j]>threshold):
                # print i,j
                # if i - j > 10:
                #     print i, j
                cons_var.append([i,j])
                cons_coef.append([1,1])
                cons_rhs.append(1)
                if not(i in ignore_list):
                    ignore_list.append(i)
                if not(j in ignore_list):                    
                    ignore_list.append(j)
    # print '   add: ' + str(len(cons_rhs)) + '  constraints, ' + str(len(ignore_list)) + '  features '
    return cons_var, cons_coef, cons_rhs, ignore_list




# # --------------------------------------------------
# # Part 2: This part resize the image to 1/4
# #      1. load the data
# #      2. build train and test 1:1 randomly
# #      3. Save train_x,y, test_x,y, B-ii, dig_E
# # --------------------------------------------------
# # Load the dataset
# print 'load the data...'
# orl_data = fetch_olivetti_faces()
# targets = orl_data.target # 400, 40 people,  10 images each
# tmp = [i*2 for i in xrange(32)]
# data = []
# for i in range(len(orl_data.images)):
#     data.append(orl_data.images[i][tmp,:][:,tmp].reshape(1024))

# data = np.array(data).T # (1024,400), 1024 = 32*32

# unique_y = np.unique(targets)
# K = len(unique_y)
# d, n = data.shape


# print 'Split the data...'

# choose_list = []
# for k in range(K):
#     choose_list.append(np.where(targets == k))
# # split the train/test data
# n_k = []
# mu_k = []
# Sigma_k = []
# for k in range(K):    
#     tmp_list = choose_list[k][0]
#     # shuffle(tmp_list)
#     tmp_n = len(tmp_list)
#     tmp_train_x = data[:,tmp_list[0:int(tmp_n/2)]]
#     tmp_train_y = targets[tmp_list[0:int(tmp_n/2)]]
#     tmp_test_x = data[:,tmp_list[int(tmp_n/2)+1:-1]]
#     tmp_test_y = targets[tmp_list[int(tmp_n/2)+1:-1]]
  
#     n_k.append(tmp_train_x.shape[1])
#     mu_k.append(np.mean(tmp_train_x,axis=1))
#     Sigma_k.append(np.cov(tmp_train_x))
    
#     if k == 0:
#         train_x = tmp_train_x
#         train_y = tmp_train_y
#         test_x = tmp_test_x
#         test_y = tmp_test_y
#     else:
#         train_x = np.concatenate([train_x, tmp_train_x],axis=1)
#         train_y = np.concatenate([train_y, tmp_train_y])
#         test_x = np.concatenate([test_x, tmp_test_x],axis=1)
#         test_y = np.concatenate([test_y, tmp_test_y])


# # calculate the two matrix
# mu = np.zeros([d])
# for k in range(K):
#     mu = mu+n_k[k]*mu_k[k] 
# mu = mu/sum(n_k)

# B = np.zeros([d,d])
# E = np.zeros([d,d])
# for k in range(K):
#     E = E + n_k[k]*Sigma_k[k]
#     B = B + n_k[k]*np.outer(mu_k[k]-mu,mu_k[k]-mu)

# dig_E = []
# dig_B = []
# B_E_ratio = []
# for i in range(d):
#     dig_B.append(B[i,i])
#     dig_E.append(E[i,i])
# dig_B = np.array(dig_B)
# dig_E = np.array(dig_E)

# print 'Save the data...'
# np.savez('orl_data_half', train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y, dig_B = dig_B, dig_E = dig_E)


if __name__ == "__main__":

    train_x, train_y, test_x, test_y, dig_B, dig_E = data_orl_random_split()
    print 'Save the data...'
    # np.savez('orl_data_all', train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y, B = B, E = E)
    np.savez('./data/orl_face/orl_data_random', train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y, dig_B = dig_B, dig_E = dig_E)
    cons_var, cons_coef, cons_rhs =  data_orl_add_constraints(0.98)
    


            


