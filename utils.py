# stdlib
import gc
import warnings
from typing import Any, Dict, Tuple

# third party
import autograd
import autograd.numpy as anp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch
from pygranso.pygransoStruct import pygransoStruct
from torch.autograd import Function

import utils

try:
    import sksparse.cholmod

    HAS_CHOLMOD = True
except ImportError:
    warnings.warn(
        "sksparse.cholmod not installed. Falling back to SciPy/SuperLU, but "
        "simulations will be about twice as slow."
    )
    HAS_CHOLMOD = False


# Default device
DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.double


def build_random_seed(seed: int):
    """
    Initialize random seeds
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mmto_design(
    design: torch.Tensor, args: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to compute the MMTO final design
    """
    num_materials = len(args['e_materials'] + 1)
    final_design = torch.zeros(args['nely'], args['nelx'], num_materials)

    for material in range(num_materials):
        final_design[:, :, material] = (
            design[:, material].reshape(args['nelx'], args['nely']).T
        )

    final_combined_material_design = np.array(
        np.argmax(final_design.numpy(), axis=2)[::-1, :]
    )
    final_np_design = np.array(final_design.detach().numpy())

    return final_combined_material_design, final_np_design


class SparseSolver(Function):
    """
    The SparseSolver method is a sparse linear solver which also
    creates the gradiends for the backward pass.
    """

    @staticmethod
    def forward(
        ctx,
        a_entries,
        a_indices,
        b,
        sym_pos=False,
        device=utils.DEFAULT_DEVICE,
        dtype=utils.DEFAULT_DTYPE,
    ):  # noqa
        # Set the inputs
        ctx.sym_pos = sym_pos
        ctx.device = device
        ctx.dtype = dtype

        # Gather the result
        a_entries_numpy = a_entries.detach().cpu().numpy()
        all_indices_numpy = a_indices.detach().cpu().numpy()
        col = all_indices_numpy.T[:, 1]
        row = all_indices_numpy.T[:, 0]
        a = scipy.sparse.csc_matrix(
            (a_entries_numpy, (row, col)),
            shape=(b.detach().cpu().numpy().size,) * 2,
        ).astype(np.float64)

        if sym_pos and HAS_CHOLMOD:
            solver = sksparse.cholmod.cholesky(a).solve_A
        else:
            # could also use scikits.umfpack.splu
            # should be about twice as slow as the cholesky
            solver = scipy.sparse.linalg.splu(a).solve

        b_np = b.data.cpu().numpy().astype(np.float64)
        result = torch.from_numpy(solver(b_np).astype(np.float64))

        # The output from the forward pass needs to have
        # requires_grad = True
        result = result.requires_grad_()

        # Need to put the result back on the same device
        if b.is_cuda:
            result = result.to(device=device, dtype=dtype)

        ctx.save_for_backward(a_entries, a_indices, result)

        del (
            a_entries,
            a_indices,
            a_entries_numpy,
            all_indices_numpy,
            row,
            col,
            solver,
            b_np,
        )
        gc.collect()
        torch.cuda.empty_cache()

        return result

    @staticmethod
    def backward(ctx, grad):  # noqa
        """
        Gather the values from the saved context
        """
        a_entries, a_indices, result = ctx.saved_tensors
        sym_pos = ctx.sym_pos
        device = ctx.device
        dtype = ctx.dtype

        # Calculate the gradient
        lambda_ = SparseSolver.apply(a_entries, a_indices, grad, sym_pos, device, dtype)
        i, j = a_indices
        i, j = i.long(), j.long()
        output = -lambda_[i] * result[j]

        del (
            a_entries,
            a_indices,
            result,
            i,
            j,
        )
        gc.collect()
        torch.cuda.empty_cache()

        return output, None, lambda_, None, None, None


def solve_coo(a_entries, a_indices, b, sym_pos, device, dtype):
    """
    Wrapper around the SparseSolver class for building
    a large sparse matrix gradient
    """
    x = SparseSolver.apply(a_entries, a_indices, b, sym_pos, device, dtype)
    return x


# Implement a find root extension for pytorch
class FindRoot(Function):
    """
    A class that extends the ability for pytorch
    to create gradients through a backward pass for solving
    the find root problem
    """

    @staticmethod
    def forward(  # type: ignore
        ctx, x, f, lower_bound, upper_bound, tolerance=1e-6
    ):  # noqa
        # define the maximum number of iterations
        max_iterations = 65

        # Save the function that will be passed into
        # this method
        ctx.function = f

        # Save the input data for the backward pass
        # For this particular case we will rely on autograd.numpy
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy().copy().astype(anp.float64)

        ctx.x_value = x

        # Start the iteration process for finding zeros
        for _ in torch.arange(max_iterations):
            y = 0.5 * (lower_bound + upper_bound)
            if (upper_bound - lower_bound) < tolerance:
                break
            if f(x, y) > 0:
                upper_bound = y
            else:
                lower_bound = y

        # Set up y to be a torch tensor with requires_grad=True
        # Needs to be set for the backward pass
        y = torch.tensor(y).requires_grad_()

        # Save the value to the ctx
        ctx.y_value = y
        return y

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        # Bring in the function that we found the zeros for
        f = ctx.function

        # Get the output variables
        # X was already set to be a regular numpy value
        x = ctx.x_value

        # In order to pass y through the function we need
        # to detach it from the graph and set it to a numpy
        # variable
        y = ctx.y_value
        if torch.is_tensor(y):
            y = y.detach().numpy().reshape((1, 1)).astype(anp.float64)

        # Set up the new functions
        g = lambda x: f(x, y)  # noqa
        h = lambda y: f(x, y)  # noqa

        # Gradient of f with respect to x
        grad_f_x = torch.from_numpy(autograd.grad(g)(x))

        # Gradient of f with respect to y
        grad_f_y = torch.from_numpy(autograd.grad(h)(y))

        # Build the gradient value
        # Adding a small constant here because I could not get the
        # gradcheck to work without this. We will want to
        # investigate later
        gradient_value = -grad_f_x / grad_f_y
        return gradient_value * grad_output, None, None, None


def find_root(x, f, lower_bound, upper_bound):
    """
    Function that wraps around the FindRoot class
    """
    return FindRoot.apply(x, f, lower_bound, upper_bound)


def sigmoid(x):
    """
    Function that builds the sigmoid function
    from autograd.numpy
    """
    return anp.tanh(0.5 * x) * 0.5 + 0.5


def logit(p):
    """
    Function that builds the logits function from
    autograd.numpy
    """
    p = anp.clip(p, 0, 1)
    return anp.log(p) - anp.log1p(-p)


def sigmoid_with_constrained_mean(x, average):
    """
    Function that will compute the sigmoid with the contrained
    mean.
    NOTE: For the udpated fuction we will use autograd.numpy
    to build f
    """
    # To avoid confusion about which variable needs to have
    # its gradient computed we will create a copy of x
    x_copy = x.detach().cpu().numpy()

    # If average is a torch tensor we need to convert it
    # to numpy
    if torch.is_tensor(average):
        average = average.numpy()

    f = lambda x, y: sigmoid(x + y).mean() - average  # noqa
    lower_bound = logit(average) - anp.max(x_copy)
    upper_bound = logit(average) - anp.min(x_copy)
    b = find_root(x, f, lower_bound, upper_bound)

    return torch.sigmoid(x + b)


def _get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist, k_entries):
    # Check if the degrees of freedom defined in the problem
    # are torch tensors
    if not torch.is_tensor(freedofs):
        freedofs = torch.from_numpy(freedofs)

    if not torch.is_tensor(fixdofs):
        fixdofs = torch.from_numpy(fixdofs)

    index_map = torch.argsort(torch.cat((freedofs, fixdofs)))

    k_xlist = k_xlist.cpu().numpy()
    k_ylist = k_ylist.cpu().numpy()
    k_entries = k_entries.detach().cpu().numpy()
    keep = (
        np.isin(k_xlist, freedofs.cpu())
        & np.isin(k_ylist, freedofs.cpu())
        & (k_entries != 0)
    )
    i = index_map[k_ylist][keep]
    j = index_map[k_xlist][keep]

    return index_map, keep, torch.stack([i, j])


def set_diff_1d(t1, t2, assume_unique=False):
    """
    Set difference of two 1D tensors.
    Returns the unique values in t1 that are not in t2.
    """
    if not assume_unique:
        t1 = torch.unique(t1)
        t2 = torch.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]


def torch_scatter1d(nonzero_values, nonzero_indices, array_len):
    """
    Function that will take a mask and non zero values and determine
    an output and ordering for the original array
    """
    all_indices = torch.arange(array_len)
    zero_indices = set_diff_1d(all_indices, nonzero_indices, assume_unique=True)
    index_map = torch.argsort(torch.cat((nonzero_indices, zero_indices)))
    values = torch.cat((nonzero_values, torch.zeros(len(zero_indices))))
    return values[index_map]


def build_node_indexes(x_phys):
    """
    Function that creates the node of the discretized grid
    for sturctural optimization
    """
    nely, nelx = x_phys.shape
    ely, elx = np.meshgrid(range(nely), range(nelx))

    # Nodes
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)
    all_ixs = np.array(
        [
            2 * n1,
            2 * n1 + 1,
            2 * n2,
            2 * n2 + 1,
            2 * n3,
            2 * n3 + 1,
            2 * n4,
            2 * n4 + 1,
        ]  # noqa
    )
    all_ixs = torch.from_numpy(all_ixs)
    return all_ixs


def get_devices():
    """
    Function to get GPU devices if available. If there are
    no GPU devices available then use CPU
    """
    # Default device
    device = torch.device("cpu")

    # Get available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        gpu_name_list = [f"cuda:{device}" for device in range(n_gpus)]

        # NOTE: For now we will only use the first GPU
        # but we might want to consider distributed GPUs in the future
        device = torch.device(gpu_name_list[0])

    return device


class HaltLog:
    """
    Save the iterations from pygranso
    """

    def haltLog(  # noqa
        self,
        iteration,
        x,
        penaltyfn_parts,
        d,
        get_BFGS_state_fn,
        H_regularized,
        ls_evals,
        alpha,
        n_gradients,
        stat_vec,
        stat_val,
        fallback_level,
    ):
        """
        Function that will create the logs from pygranso
        """
        # DON'T CHANGE THIS
        # increment the index/count
        self.index += 1

        # EXAMPLE:
        # store history of x iterates in a preallocated cell array
        self.x_iterates.append(x)
        self.f.append(penaltyfn_parts.f)
        self.tv.append(penaltyfn_parts.tv)
        self.evals.append(ls_evals)

        # keep this false unless you want to implement a custom termination
        # condition
        halt = False
        return halt

    def getLog(self):  # noqa
        """
        Once PyGRANSO has run, you may call this function to get retreive all
        the logging data stored in the shared variables, which is populated
        by haltLog being called on every iteration of PyGRANSO.
        """
        # EXAMPLE
        # return x_iterates, trimmed to correct size
        log = pygransoStruct()
        log.x = self.x_iterates[0 : self.index]
        log.f = self.f[0 : self.index]
        log.tv = self.tv[0 : self.index]
        log.fn_evals = self.evals[0 : self.index]
        return log

    def makeHaltLogFunctions(self, maxit):  # noqa
        """
        Function to make the halt log functions
        """

        # don't change these lambda functions
        def halt_log_fn(  # noqa
            iteration,
            x,
            penaltyfn_parts,
            d,
            get_BFGS_state_fn,
            H_regularized,
            ls_evals,
            alpha,
            n_gradients,
            stat_vec,
            stat_val,
            fallback_level,
        ):
            self.haltLog(
                iteration,
                x,
                penaltyfn_parts,
                d,
                get_BFGS_state_fn,
                H_regularized,
                ls_evals,
                alpha,
                n_gradients,
                stat_vec,
                stat_val,
                fallback_level,
            )

        get_log_fn = lambda: self.getLog()  # noqa

        # Make your shared variables here to store PyGRANSO history data
        # EXAMPLE - store history of iterates x_0,x_1,...,x_k
        self.index = 0
        self.x_iterates = []
        self.f = []
        self.tv = []
        self.evals = []

        # Only modify the body of logIterate(), not its name or arguments.
        # Store whatever data you wish from the current PyGRANSO iteration info,
        # given by the input arguments, into shared variables of
        # makeHaltLogFunctions, so that this data can be retrieved after PyGRANSO
        # has been terminated.
        #
        # DESCRIPTION OF INPUT ARGUMENTS
        #   iter                current iteration number
        #   x                   current iterate x
        #   penaltyfn_parts     struct containing the following
        #       OBJECTIVE AND CONSTRAINTS VALUES
        #       .f              objective value at x
        #       .f_grad         objective gradient at x
        #       .ci             inequality constraint at x
        #       .ci_grad        inequality gradient at x
        #       .ce             equality constraint at x
        #       .ce_grad        equality gradient at x
        #       TOTAL VIOLATION VALUES (inf norm, for determining feasibiliy)
        #       .tvi            total violation of inequality constraints at x
        #       .tve            total violation of equality constraints at x
        #       .tv             total violation of all constraints at x
        #       TOTAL VIOLATION VALUES (one norm, for L1 penalty function)
        #       .tvi_l1         total violation of inequality constraints at x
        #       .tvi_l1_grad    its gradient
        #       .tve_l1         total violation of equality constraints at x
        #       .tve_l1_grad    its gradient
        #       .tv_l1          total violation of all constraints at x
        #       .tv_l1_grad     its gradient
        #       PENALTY FUNCTION VALUES
        #       .p              penalty function value at x
        #       .p_grad         penalty function gradient at x
        #       .mu             current value of the penalty parameter
        #       .feasible_to_tol logical indicating whether x is feasible
        #   d                   search direction
        #   get_BFGS_state_fn   function handle to get the (L)BFGS state data
        #                       FULL MEMORY:
        #                       - returns BFGS inverse Hessian approximation
        #                       LIMITED MEMORY:
        #                       - returns a struct with current L-BFGS state:
        #                           .S          matrix of the BFGS s vectors
        #                           .Y          matrix of the BFGS y vectors
        #                           .rho        row vector of the 1/sty values
        #                           .gamma      H0 scaling factor
        #   H_regularized       regularized version of H
        #                       [] if no regularization was applied to H
        #   fn_evals            number of function evaluations incurred during
        #                       this iteration
        #   alpha               size of accepted size
        #   n_gradients         number of previous gradients used for computing
        #                       the termination QP
        #   stat_vec            stationarity measure vector
        #   stat_val            approximate value of stationarity:
        #                           norm(stat_vec)
        #                       gradients (result of termination QP)
        #   fallback_level      number of strategy needed for a successful step
        #                       to be taken.  See bfgssqpOptionsAdvanced.
        #
        # OUTPUT ARGUMENT
        #   halt                set this to true if you wish optimization to
        #                       be halted at the current iterate.  This can be
        #                       used to create a custom termination condition,
        return halt_log_fn, get_log_fn
