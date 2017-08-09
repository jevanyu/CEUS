import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_from_mat(dataset_path):
    import h5py
    if dataset_path != '':
        img = np.array(h5py.File(dataset_path)['var'])
        return img
    else:
        return None

def gaussian_kernel(kernel_size, sigma):
    kernel = np.zeros((int(kernel_size),int(kernel_size)))
    center_index_x = float(kernel_size)/2-0.5
    center_index_y = float(kernel_size)/2-0.5
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = center_index_x - i
            y = center_index_y - j
            value = (1.000/(np.sqrt(2*np.pi*float(sigma)**2)))*(math.exp(-(float(x)**2+float(y)**2)/(2*float(sigma)**2)))
            kernel[i,j] = value
    return kernel

def kernel_application(kernel, img):
    img = np.pad(img, ((int(kernel.shape[0])-1)/2), mode='constant')
    new_img = np.zeros(np.shape(img))
    print(img.shape)
    for ii in range(img.shape[0] - (kernel.shape[0]-1)):
        for jj in range(img.shape[1] - (kernel.shape[1]- 1)):
            sub_array = img[ii:ii+(kernel.shape[0]), jj:jj+(kernel.shape[1])]
            new_array = np.multiply(kernel, sub_array)
            sum_value = np.sum(new_array)
            new_img[ii,jj] = sum_value
    new_img = new_img[0:new_img.shape[0]-(int(kernel.shape[0])-1),0:new_img.shape[1]-(int(kernel.shape[0])-1)]
    return new_img

def black_background(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] < 0:
                img[i,j] = 0

# load_from_mat
img = load_from_mat("in.mat")
img = np.transpose(img)

# gaussian_kernel
kernel = gaussian_kernel(31, 3)

# kernel_application
new_img = kernel_application(kernel, img)

# make background black
final_img = black_background(new_img)

# download
fig = plt.figure(frameon=False)
fig.set_size_inches(final_img.shape[1]/10,final_img.shape[0]/10)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.gray()
ax.imshow(final_img, aspect='normal')
fig.savefig("out.png")
