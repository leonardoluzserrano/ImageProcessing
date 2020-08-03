# Most functions were written to work with grayscale images.

# Functions that can for sure be optimized:
# roll_window_filter, apply_niblack

# -------------------------------
# Libs
# -------------------------------

from PIL import Image
import numpy as np
from tqdm import tqdm
from matplotlib.pyplot import hist, show

# -------------------------------
# Functions
# -------------------------------

def open_image_as_grayscale_array(fp):
    img = Image.open(fp).convert("L")
    return np.array(img)

def save_grayscale_array_as_image(fp, mat):
    img = Image.fromarray(np.uint8(mat))
    img.save(fp)

def padding(mat, n, fill_value = 0):
    """Returns a copy of the image with zeros on the borders"""
    
    shape = int(mat.shape[0] + 2*n), int(mat.shape[1] + 2*n)
    dtype = np.uint8 if isinstance(fill_value, int) and (0 <= fill_value < 256) else None

    result = np.full(shape, fill_value, dtype)
    result[n:-n, n:-n] = mat

    return result

def correct_value_range(mat):
    """Moves and scales the range of values in the image,
    to assure that all values are between 0 and 255.
    
    Might have a numpy bug. If it does, check use_full_value_range()
    for a possible fix."""

    minimum = mat.min()
    maximum = mat.max()

    if minimum < 0:
        delta = abs(minimum)
        mat += delta
        mat = mat * 255 / (255 + delta)

    if maximum > 255:
        mat = mat * 255 / maximum

    return np.uint8(mat)

def use_full_value_range(mat):
    """Moves and scales the range of values in the image,
    to assure that darkest pixel is zero and brightest pixel is 255"""
    mat -= mat.min()
    return  np.uint8(mat / mat.max() * 255)

def roll_window_filter(mat, L):
    """
    Returns matrix obtained by applying a filter L.
    Does not implement zero padding on the borders in order to apply the filter, 
    so they are just black on the result.

    :mat: np.array of the image (only 2 dimensions, no color) 
    """
    L = L.astype(np.uint8)

    # Applying filter
    result = np.zeros(mat.shape, dtype= np.uint8)
    for y in range(1, mat.shape[0]-1):
        for x in range(1, mat.shape[1]-1):
            result[y,x] = (L * mat[y-1:y+2, x-1:x+2]).sum()

    result = correct_value_range(result)

    return result

def apply_laplace(mat, simple_window = False, sharpen_image = False, K = 0):
    """
    Can be used to get the borders in an image or make a sharpened version.
    Two types of window are available.
    """

    # Which laplace filter to use?
    if simple_window:
        L = np.array([[0,-1,0],[-1,4,-1],[-0,-1,0]])
    else:
        L = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    if sharpen_image:
        K = 1

    # Some optional tuning for the filter
    # K = 0 returns the edges
    # K = 1 returns a sharpened image
    L[1][1] += K
    
    return roll_window_filter(mat, L)

def apply_gamma(mat, gamma):
    """
    Currently, also forces use of full range of values.
    gamma < 1 ==> make brighter
    gamma > 1 ==> make darker
    """
    return use_full_value_range(mat**gamma)

def turn_binary(mat, epsilon = 3):
    # epsilon is the stop iteration threshold
    assert epsilon >= 1, "epsilon must be greater or equal to 1"

    # making sure the value range in mat is from 0 to 255
    mat = correct_value_range(mat)

    # separation threshold guess
    t = 127

    while True:
        u1 = ((mat > t) * mat).mean()   # Mean of group upper
        u2 = ((mat < t) * mat).mean()   # Mean of group lower
        new_t = (u1 + u2)/2             # New separation threshold
        diff = abs(new_t - t)
        t = new_t

        if diff < epsilon:              # If change was small, stop
            break
    
    return np.uint8(255 * (mat > t))

def invert(mat):
    return 255 - mat

def histogram_equalization(mat):
    frequency = np.array([np.count_nonzero(mat == value) for value in range(256)])
    cumulative = np.cumsum(frequency)

    mapping = cumulative / cumulative.max() * 255

    mapping_func = lambda x: mapping[x]
    elementwise_mapping_func = np.vectorize(mapping_func)

    return elementwise_mapping_func(mat).astype(np.uint8)

def plot_hist(mat):
    # Note both hist() and show() are from matplotlib
    h, _, _ = hist(mat.flatten(), bins=256)
    show()
    return h

def apply_kmeans_segmentation(mat, num_groups, epsilon = 5, verbose = False):
    # epsilon is the threshold to stop iterations

    # Defining initial centers
    centers = np.random.choice(np.unique(mat), size=num_groups, replace=False)

    # Tags
    tags = np.full(mat.shape, -1, dtype= np.int8)

    # Defining helper functions
    tag_element = lambda value: abs(value - centers).argmin()
    tag_matrix = np.vectorize(tag_element)
    calculate_center = lambda group_members: group_members.mean() if group_members.size else 0
    
    # If a center ends up with no elements at some point, what should be done?
    # Right now, it will be moved to zero.

    while True:
        # Calcule tags for all elements based on which center is closer
        tags = tag_matrix(mat)

        # Re-calculate centers as mean of the elements in that group
        new_centers = np.array([calculate_center(mat[tags == group]) for group in range(num_groups)])

        difference = abs(new_centers - centers).sum()
        centers = new_centers

        if verbose:
            print("Centers:  ", sorted(centers.astype(int)))

        # If centers barely changed, stop iteration
        if difference < epsilon:
            break

    # All elements of each group now have the same value as the center for that group
    map_element = lambda tag: centers[ tag ]
    map_matrix = np.vectorize(map_element)

    return map_matrix(tags).astype(np.uint8)

def apply_niblack(mat, n = 11, k = -0.2):
    # Algoritmo de binarização
    # k é um parametro do algoritmo

    # A implementação pode ser otimizada bastante se forem aproveitados
    # os resultados das contas da média e do std de uma janela para a próxima.

    result = np.zeros(mat.shape, dtype= np.uint8)

    # padding image with nan values,
    # so that the mean and std of the borders can be calculated
    # without python throwing "out of bounds" errors.
    pad_size = int(n/2)
    mat = padding(mat, pad_size, np.nan)
    for y in tqdm(range(pad_size, mat.shape[0]-pad_size)):
        for x in range(pad_size, mat.shape[1]-pad_size):
            window = mat[(y-pad_size):(y+1+pad_size), (x-pad_size):(x+1+pad_size)]
            m = np.nanmean(window)
            s = np.nanstd(window)
            T = m + k*s
            result[y-pad_size,x-pad_size] = 255 * (mat[y,x] >= T)

    return result

def balancedhistogramthresholding(mat):
    frequency = np.array([np.count_nonzero(mat == value) for value in range(256)])

    bot_index = 0
    top_index = 255

    while bot_index < top_index:
        T = int((bot_index + top_index)/2)

        bot_weight = frequency[bot_index:T].sum()
        top_weight = frequency[T+1:top_index].sum()

        if top_weight > bot_weight:
            top_index -= 1
        elif bot_weight > top_weight:
            bot_index += 1
        else:
            break

    return 255 * (mat > T)

def otsu(mat):
    probs = np.array([np.count_nonzero(mat == value) for value in range(256)]) / mat.size

    candidates = []
    for k in range(1,255):
        pi0 = probs[:k].sum()
        pi1 = 1 - pi0

        mu0 = (probs[:k] * range(k)).sum() / pi0
        mu1 = (probs[k:] * range(k, 256)).sum() / pi1

        mut = (probs * range(256)).sum()

        varb = pi0*pi1*(mu1-mu0)**2

        candidates.append(varb)

    T = np.argmax(candidates)

    print(T)

    return 255 * (mat > T)






# def apply_sobel_edge_detection(mat):
#     # write down L's
#     # roll window filter on them
#     # sum them up
#     # use full value range





# DIVIDIR EM CLASSES
# funcões de edgedetection
# funcoes de segmentacao
# funcoes util
# funcoes .....


# Fusão de Região Estatística (Statistical Region Merging SRM)