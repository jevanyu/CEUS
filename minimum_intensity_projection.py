import numpy as np
import matplotlib.pyplot as plt

def load_from_mat_pre(dataset_path):
    import h5py
    if dataset_path != '':
        img = np.array(h5py.File(dataset_path)['var1'])
        return img
    else:
        return None

def load_from_mat_post(dataset_path):
    import h5py
    if dataset_path != '':
        img = np.array(h5py.File(dataset_path)['var2'])
        return img
    else:
        return None

# load_from_mat_pre
img1 = load_from_mat_pre("in.mat")
img1 = np.transpose(img1)
img1 = np.min(img1,axis=2)

# load_from_mat_post
img2 = load_from_mat_post("in.mat")
img2 = np.transpose(img2)
img2 = np.min(img2,axis=2)

# create difference image
img = np.subtract(img1,img2)

# black_background
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j] < 0:
            img[i,j] = 0

# download
fig = plt.figure(frameon=False)
fig.set_size_inches(img.shape[1]/10,img.shape[0]/10)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.gray()
ax.imshow(img, aspect='normal')
fig.savefig("out.png")
