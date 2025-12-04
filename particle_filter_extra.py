import numpy as np
from particle_filter import ParticleFilter, Particle, weight_gaussian_kernel
from utils import add_noise as utils_add_noise, add_noise_laplace, add_noise_cauchy


class ParticleExtra(Particle):
    """
    Extended Particle class that supports different noise types.
    
    This class overrides add_noise to accept a noise_type parameter,
    reusing the existing logic but calling the appropriate noise function.
    """
    
    def add_noise(self, std_pos=1.0, std_orient=1.0, noise_type="gaussian"):
        """
        Adds noise to pos and orient using the specified noise distribution.
        
        This method reuses the structure from the base Particle.add_noise()
        but supports different noise types.
        
        Args:
            std_pos: standard deviation (or scale) for noise in position
            std_orient: standard deviation (or scale) for noise in orientation
            noise_type: type of noise distribution ("gaussian", "laplace", or "cauchy")
        
        Note: orient must have unit norm after adding noise
        """
        if noise_type == "gaussian":
            # Use the student's already-implemented Gaussian noise
            noise_func = utils_add_noise
            param_pos = std_pos
            param_orient = std_orient
        elif noise_type == "laplace":
            # For Laplace: Use larger scale to make heavy tails more visible
            noise_func = add_noise_laplace
            param_pos = std_pos * 1.2
            param_orient = std_orient * 1.2
        elif noise_type == "cauchy":
            # Cauchy has no variance, use smaller scale due to heavy tails
            noise_func = add_noise_cauchy
            param_pos = std_pos * 0.5
            param_orient = std_orient * 0.5
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Apply noise to position (reusing the pattern from base Particle)
        self.pos[0] = noise_func(x=self.pos[0], scale=param_pos)
        self.pos[1] = noise_func(x=self.pos[1], scale=param_pos)
        
        # Apply noise to orientation and normalize (reusing the pattern from base Particle)
        while True:
            self.orient[0] = noise_func(x=self.orient[0], scale=param_orient)
            self.orient[1] = noise_func(x=self.orient[1], scale=param_orient)
            if np.linalg.norm(self.orient) >= 1e-8:
                break
        self.orient = self.orient / np.linalg.norm(self.orient)


class ParticleFilterExtra(ParticleFilter):
    """
    EXTRA CREDIT: Extended Particle filter that supports different noise distributions
    
    This class reuses the base ParticleFilter implementation and only overrides
    the methods that need to handle different noise types.
    """
    
    def __init__(self, num_particles, minx, maxx, miny, maxy, noise_type="gaussian"):
        """
        Initialize the extended particle filter
        
        Args:
            num_particles: number of particles for this particle filter
            minx: lower bound on x-coordinate of position
            maxx: upper bound on x coordinate of position
            miny: lower bound on y coordinate of position
            maxy: upper bound on y coordinate of position
            noise_type: type of noise distribution ("gaussian", "laplace", or "cauchy")
        """
        super().__init__(num_particles, minx, maxx, miny, maxy)
        self.noise_type = noise_type
    
    def transition_sample(self, particle, delta_angle, speed):
        """
        Samples a next pose for this particle according to the car's transition model.
        Uses the noise distribution specified by self.noise_type.
        
        EXTRA CREDIT: Copy your implementation from particle_filter.py transition_sample(),
        but replace "Particle" with "ParticleExtra" and add noise_type parameter.
        """
        new_particle = None
        # BEGIN_YOUR_CODE (Extra Credit) #######################################
        # Hint: Copy your code from particle_filter.py transition_sample() method
        # Change the Particle class to ParticleExtra class and add the noise_type parameter to the add_noise method
        # For example:
        # new_particle = ParticleExtra(new_pos, new_orient)
        # new_particle.add_noise(std_pos=1.0, std_orient=0.1, noise_type=self.noise_type)
        
        raise NotImplementedError("Extra Credit: Copy your transition_sample implementation from particle_filter.py")
        
        # END_YOUR_CODE ########################################################
        return new_particle
    
    def compute_prenorm_weight(self, particle, sensor, max_sensor_range, sensor_std, evidence):
        """
        Computes the pre-normalization weight of a particle given evidence.
        Uses the appropriate kernel function based on self.noise_type.
        
        EXTRA CREDIT: Copy your implementation from particle_filter.py compute_prenorm_weight(),
        but use the appropriate kernel function based on noise_type.
        """
        weight = None
        # BEGIN_YOUR_CODE (Extra Credit) #######################################
        # Hint: Copy your code from particle_filter.py compute_prenorm_weight() method
        # and change the weight_gaussian_kernel to the appropriate kernel function based on the noise_type
        
        raise NotImplementedError("Extra Credit: Copy your compute_prenorm_weight implementation from particle_filter.py")
        
        # END_YOUR_CODE ########################################################
        return weight

def weight_laplace_kernel(x1, x2, scale=500):
    """
    EXTRA CREDIT: Returns the Laplace kernel of the distance between vectors x1 and x2
    
    The Laplace distribution PDF is: f(x) = (1/(2*scale)) * exp(-|x|/scale)
    For use as a weight, we can drop the normalization constant
    
    Args:
        x1, x2: vectors to compare
        scale: scale parameter (controls how much to penalize distance)
    
    Returns:
        weight based on Laplace kernel
    
    Hint: The Laplace distribution uses absolute distance instead of squared distance
          Compare with Gaussian kernel: exp(-distance²/(2*std)) 
          vs Laplace kernel: exp(-distance/scale)
    """
    distance = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
    return np.exp(-distance / scale)


def weight_cauchy_kernel(x1, x2, scale=500):
    """
    EXTRA CREDIT: Returns the Cauchy kernel of the distance between vectors x1 and x2
    
    The Cauchy distribution PDF is: f(x) = 1/(pi*scale*(1 + (x/scale)^2))
    For use as a weight, we can drop the normalization constant
    
    Args:
        x1, x2: vectors to compare
        scale: scale parameter (controls how much to penalize distance)
    
    Returns:
        weight based on Cauchy kernel
    
    Hint: The Cauchy kernel uses NO exponential, just polynomial decay
          Cauchy kernel: 1 / (1 + (distance/scale)²)
    """
    distance = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
    return 1.0 / (1.0 + (distance / scale) ** 2)