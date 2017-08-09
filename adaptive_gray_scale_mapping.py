import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.figure

def load_from_mat(dataset_path):
    import h5py
    if dataset_path != '':
        img = np.array(h5py.File(dataset_path)['var'])
        return img
    else:
        return None

def moments(img,radius):
    img = np.pad(img,radius,mode='constant')
    b = np.zeros(np.shape(img))
    d = np.zeros(np.shape(img))
    for i in range(img.shape[0]-radius*2):
        for j in range(img.shape[1]-radius*2):
            sub_array = img[i:i+(radius*2+1), j:j+(radius*2+1)]
            mask_b = sub_array > 0
            new_array_b = np.multiply(mask_b,sub_array)
            sum_value_b = np.sum(new_array_b)
            b[i,j] = sum_value_b
            mask_d = sub_array < 0
            new_array_d = np.multiply(mask_d,sub_array)
            sum_value_d = np.sum(new_array_d)
            d[i,j] = abs(sum_value_d)
    return b,d

def map_to_gray_level(b,d):
    new_img = np.zeros(np.shape(img))
    print("hello")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 0 and b[i,j] > d[i,j]:
                new_img[i,j] = img[i,j]*((b[i,j]-d[i,j])/b[i,j])
            elif img[i,j] < 0 and b[i,j] < d[i,j]:
                new_img[i,j] = img[i,j]*((d[i,j]-b[i,j])/d[i,j])
    return new_img

def black_background(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] < 0:
                img[i,j] = 0

# load_from_mat
img = load_from_mat("in.mat")
img = np.transpose(img)

# moments
radius = 15
b,d = moments(img,radius)

# map_to_gray_level
new_img = map_to_gray_level(b,d)

# black_background
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

# show
plt.imshow(final_img)
plt.gray()
plt.show()
