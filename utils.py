import math
import numpy as np
import sys
from PIL import Image, ImageTk

def length(x):
    return math.sqrt(x[0]**2 + x[1]**2)

def angle_bw(x, y):
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)
    if normx <= 1e-8 or normy <= 1e-8:
        return 0
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    # using cross-product formula
    return -math.degrees(math.asin((x[0] * y[1] - x[1] * y[0])/(length(x)*length(y))))
    # the dot-product formula, left here just for comparison (does not return angles in the desired range)
    # return math.degrees(math.acos((self.a * other.a + self.b * other.b)/(self.length()*other.length())))

def add_noise(x, std):
    return x + np.random.normal(0, std)

# ============================================================================
# EXTRA CREDIT: Implement these functions for Laplace and Cauchy noise
# ============================================================================

def add_noise_laplace(x, scale):
    """
    EXTRA CREDIT: Add Laplace (double exponential) noise to x
    
    The Laplace distribution has PDF: f(x) = (1/(2*scale)) * exp(-|x-mu|/scale)
    where mu is the location parameter (mean) and scale is the diversity parameter
    
    Args:
        x: the value to add noise to
        scale: the scale parameter (similar to std, but not exactly the same)
               For Laplace, variance = 2 * scale^2
    
    Returns:
        x with Laplace noise added
    
    Hint: Use np.random.laplace(loc, scale) to generate Laplace-distributed random numbers
    """
    # BEGIN_YOUR_CODE (Extra Credit) ###########################################
    raise NotImplementedError
    # END_YOUR_CODE ############################################################

def add_noise_cauchy(x, scale):
    """
    EXTRA CREDIT: Add Cauchy (Lorentz) noise to x
    
    The Cauchy distribution has PDF: f(x) = 1/(pi*scale*(1 + ((x-loc)/scale)^2))
    where loc is the location parameter and scale is the scale parameter
    
    Note: Cauchy distribution has no defined mean or variance (heavy tails!)
    
    Args:
        x: the value to add noise to
        scale: the scale parameter (controls the width of the distribution)
    
    Returns:
        x with Cauchy noise added
    
    Hint: Use np.random.standard_cauchy() and scale it appropriately
    """
    # BEGIN_YOUR_CODE (Extra Credit) ###########################################
    raise NotImplementedError
    # END_YOUR_CODE ############################################################

def load_image(path, scale):
    try:
        img = Image.open(path)
        new_width = int(img.width * float(scale))
        new_height = int(img.height * float(scale))
        img = img.resize((new_width, new_height), Image.LANCZOS)
        return img, ImageTk.PhotoImage(img)
    except IOError as e:
        print(e)
        sys.exit(1)