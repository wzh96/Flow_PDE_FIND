import torch
from scipy.special import binom
def build_collection_library(z, poly_order, include_sine = False):
    """
            Arguments:
                z - 2D pytorch tensor array of the snapshots on which to build the library. Shape is the number of time
                points by the number of state variables.
                lantent_dim - Integer, number of state variables in z.
                poly_order - Integer, polynomial order to which to build the library, max is 5
                include_sine: Boolean, whether to include sine terms in the library

            Returns:
                 tensorflow array containing the constructed library. Shape is number of time points
                 number of library functions. The number of library functions is determined by the number
                 of state variables of the input, the polynomial order, and whether sines are included.
    """
    dim = z.size(1)
    library = [torch.ones(z.size()[0])]
    for i in range(dim):
        library.append(z[:,i])
    if poly_order > 1:
        for i in range(dim):
            for j in range(i, dim):
                library.append(z[:,i]*z[:,j])
    if poly_order > 2:
        for i in range(dim):
            for j in range(i, dim):
                for k in range(j,dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])
    if poly_order > 3:
        for i in range(dim):
            for j in range(i,dim):
                for k in range(j,dim):
                    for p in range(k,dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])
    if include_sine:
        for i in range(dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, dim=1)

def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l