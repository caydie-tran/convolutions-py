from skimage import io, color, viewer
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import scipy.signal
import pylab

def convolve_2d_kernel(image_arr, kernel):
    # Flip the kernel both horizontally and vertically.
    kernel = np.flipud(np.fliplr(kernel))
    # Initialize a zero array with same dimension as the given image array.
    result = np.zeros_like(image_arr)

    # Pad the array with zeros on each side.
    padded_rows = image_arr.shape[0] + 2
    padded_columns = image_arr.shape[1] + 2

    padded_arr = np.zeros((padded_rows, padded_columns))
    
    # Let all pixels not in the padded edges equal the given image array.
    padded_arr[1: -1, 1: -1] = image_arr
    
    # Mutliply every element in the kernel with corresponding element in the 
    # image array patch and then take the sum of the products.
    for i in range(image_arr.shape[1]):
        for j in range(image_arr.shape[0]):
            patch = padded_arr[i: i + 3, j: j + 3]
            result[i, j] = np.sum(kernel * patch)
    return result

image = io.imread('flowers.jpg') # Read image.
image = color.rgb2gray(image) # Transform image from 3 color channel to 1 color channel.

# SHARPEN KERNEL 
# Offset contrast loss from zero padding.
# A higher clip_limit value means more contrast.
equalized_image = exposure.equalize_adapthist(image/np.max(np.abs(image)), clip_limit=0.01)
plt.imshow(equalized_image, cmap=plt.cm.gray)
plt.axis('off') # Remove axis from plot.
plt.show()

# Convolve the sharpen kernel and the image
sharpen_kernel = np.array([[-1, -1, -1],[-1, 8,-1],[-1,-1,-1]])
image_sharpened = convolve_2d_kernel(image, sharpen_kernel)

# Adjust the contrast of the filtered image by applying Histogram Equalization 
image_sharpened_equalized = exposure.equalize_adapthist(image_sharpened/np.max(np.abs(image_sharpened)), clip_limit=0.01)
plt.imshow(image_sharpened_equalized, cmap=plt.cm.gray)
plt.axis('off') # Remove axis from plot.
plt.show()


# DENOISED EDGE DETECTION USING scipy.signal.convolve2d
# Apply sharpen kernel.
image_sharpened = scipy.signal.convolve2d(image, sharpen_kernel, 'valid')
# Apply edge kernel.
edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
edge_detection_image = scipy.signal.convolve2d(image_sharpened, edge_kernel, 'valid')
# Apply box blur kernel to remove noise.
box_blur_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/9.0
denoised_image = scipy.signal.convolve2d(edge_detection_image, box_blur_kernel, 'valid')
denoised_equalized_image = exposure.equalize_adapthist(denoised_image/np.max(np.abs(denoised_image)), clip_limit=0.01)
plt.imshow(denoised_equalized_image, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

