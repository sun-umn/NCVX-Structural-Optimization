# stdlib
import gc
from typing import Any, Dict, List

# third party
import numpy as np
import pandas as pd
import torch
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

# first party
import models
import topo_api
import topo_physics
import utils


# Incorporating Multi-Material Designs
def multi_material_constraint_function(
    model, initial_compliance, ke, args, add_constraints, device, dtype
):
    """
    Function to implement MMTO constraints for our framework
    """
    model.eval()

    # TODO: Changing to V2
    (
        unscaled_compliance,
        x_phys,
        mask,
    ) = topo_physics.calculate_multi_material_compliance(model, ke, args, device, dtype)

    model.train()
    f = 1.0 / initial_compliance * unscaled_compliance

    # Run this problem with no inequality constraints
    ci = None

    ce = pygransoStruct()
    num_materials = len(args['e_materials'])
    total_mass = (
        torch.max(args['material_density_weight'])
        * args['nelx']
        * args['nely']
        * args['combined_frac']
    )
    # Pixel total here will be the number of rows
    # pixel_total = x_phys.shape[0]
    pixel_total = x_phys.numel()

    mass_constraint = torch.zeros(num_materials)
    for index, density_weight in enumerate(args['material_density_weight']):
        mass_constraint[index] = density_weight * torch.sum(x_phys[:, index + 1])

    c1 = (torch.sum(mass_constraint) / total_mass) - 1.0
    ce.c1 = c1  # noqa

    # TODO: Remove? if add_constraints:
    # Directly binary constraint
    binary_constraint = x_phys * (1 - x_phys)
    c2 = torch.norm(binary_constraint, p=1) / pixel_total
    ce.c2 = c2

    print(unscaled_compliance.item(), c1.item(), c2.item())

    # Let's try and clear as much stuff as we can to preserve memory
    del x_phys, mask, ke
    gc.collect()
    torch.cuda.empty_cache()

    return f, ci, ce


def multi_material_constraint_function_v2(
    model,
    initial_compliance,
    ke,
    args,
    volume_constraint_list,
    binary_constraint_list,
    symmetry_constraint_list,
    iter_counter,
    device,
    dtype,
):
    """
    Function to implement MMTO constraints for our framework
    """
    model.eval()

    # TODO: Changing to V2
    (
        unscaled_compliance,
        x_phys,
        mask,
    ) = topo_physics.calculate_multi_material_compliance_v2(
        model, ke, args, device, dtype
    )

    model.train()
    f = 1.0 / initial_compliance * unscaled_compliance

    # Run this problem with no inequality constraints
    ci = None

    ce = pygransoStruct()
    num_materials = len(args['e_materials'])
    total_mass = (
        torch.max(args['material_density_weight'])
        * args['nelx']
        * args['nely']
        * args['combined_frac']
    )

    # Pixel total here will be the number of rows
    pixel_total = x_phys.numel()

    mass_constraint = torch.zeros(num_materials)
    for index, density_weight in enumerate(args['material_density_weight']):
        mass_constraint[index] = density_weight * torch.sum(x_phys[index + 1, :, :])

    c1 = (torch.sum(mass_constraint) / total_mass) - 1.0
    volume_constraint_list.append(np.abs(c1.item()))
    ce.c1 = c1  # noqa

    # Binary constraint
    binary_constraint = x_phys * (1 - x_phys)
    c2 = torch.norm(binary_constraint, p=1) / pixel_total
    binary_constraint_list.append(np.abs(c2.item()))
    ce.c2 = c2

    print(unscaled_compliance.item(), c1.item(), c2.item())

    iter_counter += 1

    # Let's try and clear as much stuff as we can to preserve memory
    del x_phys, mask, ke
    gc.collect()
    torch.cuda.empty_cache()

    return f, ci, ce


def train_pygranso_mmto(model, comb_fn, maxit, device) -> None:
    # Initalize the opts
    opts = pygransoStruct()

    # Set the device
    opts.torch_device = device

    # Setup the intitial inputs for the solver
    nvar = getNvarTorch(model.parameters())
    opts.x0 = (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .reshape(nvar, 1)
    ).to(device=device, dtype=torch.double)

    # Additional pygranso options
    opts.limited_mem_size = 100
    opts.torch_device = device
    opts.double_precision = True
    opts.mu0 = 1.0
    opts.maxit = maxit
    opts.print_frequency = 1
    opts.stat_l2_model = False
    opts.viol_eq_tol = 1e-4
    opts.opt_tol = 1e-4

    # Main algorithm with logging enabled.
    _ = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)


# Updated Section for training PyGranso with Direct Volume constraints
# TODO: We will want to create a more generalized format for training our PyGranso
# problems for for now we will have a separate training process for the volume
# constraint that we have been working on
# Volume constrained function
def volume_constrained_structural_optimization_function(
    model,
    initial_compliance,
    ke,
    args,
    volume_constraint_list,
    binary_constraint_list,
    symmetry_constraint_list,
    iter_counter,
    trial_index,
    device,
    dtype,
):
    """
    Combined function for PyGranso for the structural optimization
    problem. The inputs will be the model that reparameterizes x as a function
    of a neural network. V0 is the initial volume, K is the global stiffness
    matrix and F is the forces that are applied in the problem.

    Notes:
    For the original MBB Beam the best alpha is 5e3
    """
    # Initialize the model
    # In my version of the model it follows the similar behavior of the
    # tensorflow repository and only needs None to initialize and output
    # a first value of x

    unscaled_compliance, x_phys, mask = topo_physics.calculate_compliance(
        model, ke, args, device, dtype
    )
    f = 1.0 / initial_compliance * unscaled_compliance

    # Run this problem with no inequality constraints
    ci = None

    ce = pygransoStruct()
    # Directly handle the volume contraint
    total_elements = x_phys[mask].numel()
    volume_constraint = (torch.mean(x_phys[mask]) / args['volfrac']) - 1.0
    ce.c1 = volume_constraint  # noqa

    # Directly handle the binary constraint
    epsilon = args["epsilon"]
    binary_constraint = (
        torch.norm(x_phys[mask] * (1 - x_phys[mask]), p=1) / total_elements
    ) - epsilon
    ce.c2 = binary_constraint

    # What if we enforce a symmetry constraint as well?
    midpoint = x_phys.shape[0] // 2
    x_phys_top = x_phys[:midpoint, :]
    x_phys_bottom = x_phys[midpoint:, :]

    # For this part we will need to ignore the mask for now
    if symmetry_constraint_list is not None:
        symmetry_constraint = (
            torch.norm(x_phys_top - torch.flip(x_phys_bottom, [0]), p=1)
            / total_elements
        ) - epsilon
        ce.c3 = symmetry_constraint

        # Add the data to the list
        symmetry_constraint_value = symmetry_constraint
        symmetry_constraint_value = float(
            symmetry_constraint_value.detach().cpu().numpy()
        )
        symmetry_constraint_list.append(symmetry_constraint_value)

    # We need to save the information from the trials about volume
    volume_constraint_list.append(volume_constraint.detach().cpu().numpy())

    # Binary constraint
    binary_constraint_list.append(binary_constraint.detach().cpu().numpy())

    # Update the counter by one
    iter_counter += 1

    # # Let's try and clear as much stuff as we can to preserve memory
    del x_phys, mask, ke
    gc.collect()
    torch.cuda.empty_cache()

    return f, ci, ce


def train_pygranso(
    problem,
    pygranso_combined_function,
    device,
    requires_flip,
    total_frames,
    cnn_kwargs=None,
    *,
    num_trials=50,
    mu=1.0,
    maxit=500,
    epsilon=1e-3,
    include_symmetry=False,
    include_penalty=False
) -> Dict[str, Any]:
    """
    Function to train structural optimization pygranso
    """
    # Set up the dtypes
    dtype32 = torch.double
    default_dtype = utils.DEFAULT_DTYPE

    # Get the problem args
    args = topo_api.specified_task(problem, device=device)
    if not include_penalty:
        args['penal'] = 3.0

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"],
        poisson=args["poisson"],
        device=device,
    )

    # Trials
    trials_designs = np.zeros((num_trials, args["nely"], args["nelx"]))
    trials_losses = np.full((maxit + 1, num_trials), np.nan)
    trials_volumes = np.full((maxit + 1, num_trials), np.nan)
    trials_binary_constraint = np.full((maxit + 1, num_trials), np.nan)
    trials_symmetry_constraint = np.full((maxit + 1, num_trials), np.nan)
    trials_initial_volumes = []

    for index, seed in enumerate(range(0, num_trials)):
        models.set_seed(seed * 10)
        counter = 0
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize the CNN Model
        if cnn_kwargs is not None:
            cnn_model = models.MultiMaterialCNNModel(
                args, random_seed=seed, **cnn_kwargs
            ).to(device=device, dtype=torch.double)
            cnn_model = cnn_model.to(device=device, dtype=dtype32)
        else:
            cnn_model = models.MultiMaterialCNNModel(args, random_seed=seed).to(  # type: ignore  # noqa
                device=device, dtype=dtype32
            )

        # Calculate initial compliance
        cnn_model.eval()
        with torch.no_grad():
            (
                initial_compliance,
                init_x_phys,
                init_mask,
            ) = topo_physics.calculate_compliance(
                cnn_model, ke, args, device, default_dtype
            )

        # Get the initial compliance
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1e-2
        )

        # Get the initial volume
        initial_volume = torch.mean(init_x_phys[init_mask])
        trials_initial_volumes.append(initial_volume.detach().cpu().numpy())

        # Put the cnn model in training mode
        cnn_model.train()

        # Combined function
        volume_constraint: List[float] = []  # noqa
        binary_constraint: List[float] = []  # noqa

        symmetry_constraint = None
        if include_symmetry:
            symmetry_constraint = []  # type: ignore

        comb_fn = lambda model: pygranso_combined_function(  # noqa
            cnn_model,  # noqa
            initial_compliance,
            ke,
            args,
            volume_constraint_list=volume_constraint,  # noqa
            binary_constraint_list=binary_constraint,  # noqa
            symmetry_constraint_list=symmetry_constraint,  # noqa
            iter_counter=counter,
            trial_index=index,
            device=device,
            dtype=default_dtype,
        )

        # Initalize the pygranso options
        opts = pygransoStruct()

        # Set the device
        opts.torch_device = device

        # Setup the intitial inputs for the solver
        nvar = getNvarTorch(cnn_model.parameters())
        opts.x0 = (
            torch.nn.utils.parameters_to_vector(cnn_model.parameters())
            .detach()
            .reshape(nvar, 1)
        ).to(device=device, dtype=dtype32)

        # Additional pygranso options
        opts.limited_mem_size = 20
        opts.torch_device = device
        opts.double_precision = True
        opts.mu0 = mu
        opts.maxit = maxit
        opts.print_frequency = 20
        opts.stat_l2_model = False
        opts.viol_eq_tol = 1e-7
        opts.opt_tol = 1e-7

        mHLF_obj = utils.HaltLog()
        halt_log_fn, get_log_fn = mHLF_obj.makeHaltLogFunctions(opts.maxit)

        #  Set PyGRANSO's logging function in opts
        opts.halt_log_fn = halt_log_fn

        # Main algorithm with logging enabled.
        soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)

        # GET THE HISTORY OF ITERATES
        # Even if an error is thrown, the log generated until the error can be
        # obtained by calling get_log_fn()
        log = get_log_fn()

        # Final structure
        indexes = (pd.Series(log.fn_evals).cumsum() - 1).values.tolist()

        cnn_model.eval()
        with torch.no_grad():
            final_compliance, final_design, _ = topo_physics.calculate_compliance(
                cnn_model, ke, args, device, default_dtype
            )
            final_design = final_design.detach().cpu().numpy()

        # Calculate metrics on original scale
        final_f = soln.final.f * initial_compliance.cpu().numpy()
        log_f = pd.Series(log.f) * initial_compliance.cpu().numpy()

        # Save the data from each trial
        # trials
        trials_designs[index, :, :] = final_design
        trials_losses[: len(log_f), index] = log_f.values  # noqa

        volume_constraint_arr = np.asarray(volume_constraint)
        trials_volumes[: len(log_f), index] = volume_constraint_arr[indexes]

        binary_constraint_arr = np.asarray(binary_constraint)
        trials_binary_constraint[: len(log_f), index] = binary_constraint_arr[indexes]

        if include_symmetry:
            symmetry_constraint_arr = np.asarray(symmetry_constraint)
            trials_symmetry_constraint[: len(log_f), index] = symmetry_constraint_arr[
                indexes
            ]

        # Remove all variables for the next round
        del (
            cnn_model,
            comb_fn,
            opts,
            mHLF_obj,
            halt_log_fn,
            get_log_fn,
            soln,
            log,
            final_design,
            final_f,
            log_f,
            volume_constraint,
            binary_constraint,
            symmetry_constraint,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # If symmetry is NOT included then the array will just
    # be nan
    outputs = {
        "designs": trials_designs,
        "losses": trials_losses,
        "volumes": trials_volumes,
        "binary_constraint": trials_binary_constraint,
        "symmetry_constraint": trials_symmetry_constraint,
        # Convert to numpy array
        "trials_initial_volumes": np.array(trials_initial_volumes),
    }

    return outputs


def train_pygranso_v2(
    problem,
    pygranso_combined_function,
    device,
    requires_flip,
    total_frames,
    cnn_kwargs=None,
    *,
    num_trials=50,
    mu=1.0,
    maxit=500,
    include_symmetry=False,
    include_penalty=False
) -> Dict[str, Any]:
    """
    Function to train structural optimization pygranso
    """
    # Set up the dtypes
    dtype32 = torch.double
    default_dtype = utils.DEFAULT_DTYPE

    # Get the problem args
    args = topo_api.specified_task(problem, device=device)
    if not include_penalty:
        args['penal'] = 3.0

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"],
        poisson=args["poisson"],
        device=device,
    )

    # Trials
    trials_designs = np.zeros((num_trials, args["nely"], args["nelx"]))
    trials_losses = np.full((maxit + 1, num_trials), np.nan)
    trials_volumes = np.full((maxit + 1, num_trials), np.nan)
    trials_binary_constraint = np.full((maxit + 1, num_trials), np.nan)
    trials_symmetry_constraint = np.full((maxit + 1, num_trials), np.nan)
    trials_initial_volumes = []

    for index, seed in enumerate(range(0, num_trials)):
        models.set_seed(seed * 10)
        counter = 0
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize the CNN Model
        if cnn_kwargs is not None:
            cnn_model = models.MultiMaterialCNNModelV2(
                args, random_seed=seed, **cnn_kwargs
            ).to(device=device, dtype=torch.double)
            cnn_model = cnn_model.to(device=device, dtype=dtype32)
        else:
            cnn_model = models.MultiMaterialCNNModelV2(args, random_seed=seed).to(  # type: ignore  # noqa
                device=device, dtype=dtype32
            )

        # Calculate initial compliance
        cnn_model.eval()
        with torch.no_grad():
            (
                initial_compliance,
                init_x_phys,
                init_mask,
            ) = topo_physics.calculate_multi_material_compliance_v2(
                cnn_model, ke, args, device, default_dtype
            )

        # Get the initial compliance
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1e-2
        )

        # Get the initial volume
        initial_volume = torch.mean(init_x_phys[init_mask])
        trials_initial_volumes.append(initial_volume.detach().cpu().numpy())

        # Put the cnn model in training mode
        cnn_model.train()

        # Combined function
        volume_constraint: List[float] = []  # noqa
        binary_constraint: List[float] = []  # noqa

        symmetry_constraint = None
        if include_symmetry:
            symmetry_constraint = []  # type: ignore

        comb_fn = lambda model: pygranso_combined_function(  # noqa
            cnn_model,  # noqa
            initial_compliance,
            ke,
            args,
            volume_constraint_list=volume_constraint,  # noqa
            binary_constraint_list=binary_constraint,  # noqa
            symmetry_constraint_list=symmetry_constraint,  # noqa
            iter_counter=counter,
            device=device,
            dtype=default_dtype,
        )

        # Initalize the pygranso options
        opts = pygransoStruct()

        # Set the device
        opts.torch_device = device

        # Setup the intitial inputs for the solver
        nvar = getNvarTorch(cnn_model.parameters())
        opts.x0 = (
            torch.nn.utils.parameters_to_vector(cnn_model.parameters())
            .detach()
            .reshape(nvar, 1)
        ).to(device=device, dtype=dtype32)

        # Additional pygranso options
        opts.limited_mem_size = 20
        opts.torch_device = device
        opts.double_precision = True
        opts.mu0 = mu
        opts.maxit = maxit
        opts.print_frequency = 1
        opts.stat_l2_model = False
        opts.viol_eq_tol = 1e-7
        opts.opt_tol = 1e-7

        mHLF_obj = utils.HaltLog()
        halt_log_fn, get_log_fn = mHLF_obj.makeHaltLogFunctions(opts.maxit)

        #  Set PyGRANSO's logging function in opts
        opts.halt_log_fn = halt_log_fn

        # Main algorithm with logging enabled.
        soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)

        # GET THE HISTORY OF ITERATES
        # Even if an error is thrown, the log generated until the error can be
        # obtained by calling get_log_fn()
        log = get_log_fn()

        # Final structure
        indexes = (pd.Series(log.fn_evals).cumsum() - 1).values.tolist()

        cnn_model.eval()
        with torch.no_grad():
            (
                final_compliance,
                final_design,
                _,
            ) = topo_physics.calculate_multi_material_compliance_v2(
                cnn_model, ke, args, device, default_dtype
            )
            final_design = final_design.detach().cpu().numpy()

        # Calculate metrics on original scale
        final_f = soln.final.f * initial_compliance.cpu().numpy()
        log_f = pd.Series(log.f) * initial_compliance.cpu().numpy()

        # Save the data from each trial
        # trials
        trials_designs[index, :, :] = final_design.argmax(axis=0)
        trials_losses[: len(log_f), index] = log_f.values  # noqa

        volume_constraint_arr = np.asarray(volume_constraint)
        trials_volumes[: len(log_f), index] = volume_constraint_arr[indexes]

        binary_constraint_arr = np.asarray(binary_constraint)
        trials_binary_constraint[: len(log_f), index] = binary_constraint_arr[indexes]

        if include_symmetry:
            symmetry_constraint_arr = np.asarray(symmetry_constraint)
            trials_symmetry_constraint[: len(log_f), index] = symmetry_constraint_arr[
                indexes
            ]

        # Remove all variables for the next round
        del (
            cnn_model,
            comb_fn,
            opts,
            mHLF_obj,
            halt_log_fn,
            get_log_fn,
            soln,
            log,
            final_design,
            final_f,
            log_f,
            volume_constraint,
            binary_constraint,
            symmetry_constraint,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # If symmetry is NOT included then the array will just
    # be nan
    outputs = {
        "designs": trials_designs,
        "losses": trials_losses,
        "volumes": trials_volumes,
        "binary_constraint": trials_binary_constraint,
        "symmetry_constraint": trials_symmetry_constraint,
        # Convert to numpy array
        "trials_initial_volumes": np.array(trials_initial_volumes),
    }

    return outputs


def unconstrained_structural_optimization_function(model, ke, args, designs, losses):
    """
    Combined function for PyGranso for the structural optimization
    problem. The inputs will be the model that reparameterizes x as a function
    of a neural network. V0 is the initial volume, K is the global stiffness
    matrix and F is the forces that are applied in the problem.
    """
    # Initialize the model
    # In my version of the model it follows the similar behavior of the
    # tensorflow repository and only needs None to initialize and output
    # a first value of x
    logits = model(None)

    # kwargs for displacement
    kwargs = dict(
        penal=torch.tensor(args["penal"]),
        e_min=torch.tensor(args["young_min"]),
        e_0=torch.tensor(args["young"]),
    )

    # Calculate the physical density
    x_phys = topo_physics.physical_density(logits, args, volume_constraint=True)  # noqa

    # Calculate the forces
    forces = topo_physics.calculate_forces(x_phys, args)

    # Calculate the u_matrix
    u_matrix, _ = topo_physics.sparse_displace(
        x_phys, ke, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    # Calculate the compliance output
    compliance_output = topo_physics.compliance(x_phys, u_matrix, ke, **kwargs)

    # The loss is the sum of the compliance
    f = torch.sum(compliance_output)

    # Run this problem with no inequality constraints
    ci = None

    # Run this problem with no equality constraints
    ce = None

    # Append updated physical density designs
    designs.append(
        topo_physics.physical_density(logits, args, volume_constraint=True)
    )  # noqa

    return f, ci, ce
