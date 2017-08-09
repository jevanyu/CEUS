import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def load_from_mat1(dataset_path):
    import h5py
    if dataset_path != '':
        img = np.array(h5py.File(dataset_path)['roiB'])
        return img
    else:
        return None

def load_from_mat2(dataset_path):
    import h5py
    if dataset_path != '':
        img = np.array(h5py.File(dataset_path)['roiT'])
        return img
    else:
        return None

img_b = load_from_mat1('in.mat')
img_b = np.transpose(img_b)

img_t = load_from_mat2('in.mat')
img_t = np.transpose(img_t)

diff_image = cv2.imread('in.png')
diff_image = np.mean(diff_image, axis=2)

apply_b = np.multiply(diff_image,img_b)
apply_t = np.multiply(diff_image,img_t)

s_bubble = 0
for i in range(apply_b.shape[0]):
	for j in range(apply_b.shape[1]):
		if apply_b[i,j] != 0:
			s_bubble += apply_b[i,j] ** 2

n_b = np.sum(img_b)

ms_bubble = float(s_bubble/n_b)
rms_bubble = np.sqrt(ms_bubble)

s_tissue = 0
for i in range(apply_t.shape[0]):
	for j in range(apply_t.shape[1]):
		if apply_t[i,j] != 0:
			s_tissue += apply_t[i,j] ** 2

n_t = np.sum(img_t)

ms_tissue = float(s_tissue/n_t)
rms_tissue = np.sqrt(ms_tissue)

snr = 20.0*np.log10(rms_bubble/rms_tissue)

print(snr)
