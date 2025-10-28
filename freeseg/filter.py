import sys
import numpy as np
import torch
import logging

class Filter:
    @staticmethod
    def gaussian_kernel_1d(sigma, max_sigma=None, truncate=2.5, radius=None, device=None, dtype=None):
        """
        generate a 1D Gaussian kernel with given standard deviation

        sigma: float
          Standard deviation for Gaussian kernel.
        truncate: float, optional
          Truncate the filter at this many standard deviations. Default is 2.5.
        radius: None or int, optional
          Radius of the Gaussian kernel. 
          If specified, the size of the kernel along each axis will be 2*radius + 1, 
          and truncate is ignored. Default is None
        """

        if (radius is None):
            if (max_sigma is None):
                max_sigma = sigma
            radius = int(np.ceil(truncate * max_sigma) / 2)

        # calculate the kernel range
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)

        """
        The 1D gaussian pdf for zero-mean is given as
            (1 / (sigma * sqrt(2*pi))) * np.exp(-(x**2) / (2*sigma^2))
        The constant term (1 / (sigma * sqrt(2*pi))) will be cancelled out when we normalize the discrete kernel.
        """
        pdf = torch.exp(-(x**2) / (2 * sigma**2))
        
        # return ormalized the kernel
        return pdf / pdf.sum()


    @staticmethod
    def gaussian_kernel(sigma, max_sigma=None, truncate=2.5, radius=None, device=None, dtype=None, separable=False):
        """
        ??? todo: implement separable=True ???
        ??? todo: handle sigma = 0 ???
        ??? Q. for sigma = 0, the kernel has all zeros except the middle value = 1 ???

        generate a Gaussian kernel with given standard deviation        

        sigma: list
          Standard deviation for Gaussian kernel. The standard deviations are given for each axis as a list.
        truncate: float, optional
          Truncate the filter at this many standard deviations. Default is 2.5.
        radius: None, int, or list, optional
          Radius of the Gaussian kernel. The radius are given for each axis as a list, 
          or as a single number, in which case it is equal for all axes.
          If specified, the size of the kernel along each axis will be 2*radius + 1, 
          and truncate is ignored. Default is None.
        """

        ndims = len(sigma)

        if (radius is None):
            # compute the radii of the kernel for each dimension
            if (max_sigma is None):
                max_sigma = sigma
            elif (np.isscalar(max_sigma)):
                max_sigma = [max_sigma] * ndims
            radius = [int(np.ceil(truncate * s) / 2) for s in max_sigma]
        elif (np.isscalar(radius)):
            radius = [radius] * ndims

        assert (ndims == len(radius)), \
            f"freeseg.filter.Filter.gaussian_kernel(): sigma and radius need to be the same length"
        
        # generate a range of indices for each dimension
        ranges = [torch.arange(-r, r + 1, device=device) for r in radius]

        # create a meshgrid of indices for all dimensions [k1, k2, k3, ndims] 
        grid = torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=-1)

        # shape of the kernel [k1, k2, k3]
        kernel_shape = grid.shape[:-1]

        # convert the standard deviations to a tensor
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)

        """
        The 3D gaussian pdf for zero-mean is given as
            constant_term * np.exp(-(x^2+y^2+z^2) / (2*sigma^2))
            where constant_term = 1 / ((sigma^3) * ((2*pi)^(3/2)))
        The constant_term will be cancelled out when we normalize the discrete kernel.
        
        The 'sum(-1)' in the pdf calculation below is to sum up all 3 dimensions x, y, z.
        pdf is first computed in shape [k1xk2xk3], then reshaped into [k1, k2, k3].
        """
        # reshape the grid into [k1xk2xk3, ndims]
        grid = grid.view(-1, ndims)        
        #pdf = torch.exp(-((grid**2).sum(-1) / (2 * sigma**2))).view(kernel_shape)
        pdf = torch.exp(-((grid**2) / (2 * sigma**2)).sum(-1)).view(kernel_shape)

        # return normalized the kernel
        return pdf / pdf.sum()

