# stdlib
import gc
from typing import Any, Dict, List, Optional

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


def multi_material_constraint_function(
    model, initial_compliance, ke, args, device, dtype
):
    """
    Function to implement MMTO constraints for our framework
    """
    model.eval()

    # Calculate compliance and designs
    (
        unscaled_compliance,
        x_phys,
        logits,
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
    pixel_total = args['nelx'] * args['nely']

    mass_constraint = torch.zeros(num_materials)
    for index, density_weight in enumerate(args['material_density_weight']):
        mass_constraint[index] = density_weight * torch.sum(logits[index + 1, :, :])

    c1 = (torch.sum(mass_constraint) / total_mass) - 1.0
    ce.c1 = c1  # noqa

    binary_constraint = 0
    bcv = 0
    for i in range(num_materials):
        channel = x_phys[:, i + 1].flatten()
        binary_constraint_value = torch.norm(channel * (1 - channel), p=1)
        bcv += binary_constraint_value.detach().item()
        binary_constraint += binary_constraint_value / pixel_total

    ce.c2 = binary_constraint - 5e-4

    # Add symmetry
    x_symmetry = 0
    if args["x_symmetry"]:
        x_size = logits.shape[2]
        midpoint = x_size // 2
        for i in range(num_materials + 1):
            channel = logits[i, :, :]
            channel_l = channel[:midpoint, :]
            channel_r = torch.flip(channel[-midpoint:, :], [1])
            symmetry_constraint = torch.norm(channel_l - channel_r, p=1) / pixel_total
            x_symmetry += symmetry_constraint

        ce.c3 = x_symmetry

    print(
        unscaled_compliance.item(),
        c1.item(),
        binary_constraint.item(),
        x_symmetry,
        bcv,
    )

    # Let's try and clear as much stuff as we can to preserve memory
    del x_phys, mask, ke
    gc.collect()
    torch.cuda.empty_cache()

    return f, ci, ce


def multi_material_constraint_function_v2(
    model,
    initial_compliance,
    initial_void_compliance,
    ke,
    args,
    compliance_list,
    volume_constraint_list,
    discrete_constraint_list,
    x_symmetry_constraint_list,
    y_symmetry_constraint_list,
    device,
    dtype,
):
    """
    Function to implement MMTO constraints for our framework
    """
    # Variables
    num_materials = len(args['e_materials']) + 1
    total_mass = (
        torch.max(args['material_density_weight'])
        * args['nelx']
        * args['nely']
        * args['combined_frac']
    )
    pixel_total = args['nelx'] * args['nely']
    material_density_weight = args['material_density_weight']

    # Calculate compliance and designs
    (
        compliance,
        x_phys,
        logits,
        mask,
    ) = topo_physics.calculate_multi_material_compliance(model, ke, args, device, dtype)

    # Compute the void compliance
    (
        void_compliance,
        _,
        _,
    ) = topo_physics.calculate_void_compliance(model, ke, args, device, dtype)

    # Model in training mode
    model.train()

    # Pygranso setup
    f = 1.0 / initial_compliance * compliance
    f_void = 1.0 / initial_void_compliance * void_compliance

    ce = pygransoStruct()
    ci = None

    # Compute the mass constraint
    mass_constraint = torch.zeros(num_materials - 1)
    for index, density_weight in enumerate(material_density_weight):
        mass_constraint[index] = density_weight * torch.sum(x_phys[:, index + 1])

    mass_constraint = (torch.sum(mass_constraint) / total_mass) - 1.0
    ce.c1 = mass_constraint  # noqa

    # Compute the discrete (binary) constraint
    epsilon = args["epsilon"]
    discrete_constraint = 0
    for i in range(num_materials):
        material_channel_values = x_phys[:, i + 1].flatten()
        discrete_constraint_value = torch.norm(
            material_channel_values * (1 - material_channel_values), p=1
        )
        discrete_constraint += discrete_constraint_value / pixel_total

    discrete_constraint = discrete_constraint - epsilon
    ce.c2 = discrete_constraint

    # Compute symmetry values
    x_symmetry_constraint = torch.tensor(0.0)
    if args["x_symmetry"]:

        # Get the midpoint of the design grid
        x_size = logits.shape[2]
        midpoint = x_size // 2

        # Each material channel should have symmetry
        for i in range(num_materials):
            channel_values = logits[i, :, :]
            channel_left = channel_values[:, :midpoint]
            channel_right = torch.flip(channel_values[:, -midpoint:], [1])
            x_symmetry_constraint_value = (
                torch.norm(channel_left - channel_right, p=1) / pixel_total
            )
            x_symmetry_constraint += x_symmetry_constraint_value

        ce.c3 = x_symmetry_constraint

    y_symmetry_constraint = torch.tensor(0.0)
    if args["y_symmetry"]:

        # Get the midpoint of the design grid
        y_size = logits.shape[1]
        midpoint = y_size // 2

        # Each material channel should have symmetry
        for i in range(num_materials):
            channel_values = logits[i, :, :]
            channel_top = channel_values[:midpoint, :]
            channel_bottom = torch.flip(channel_values[-midpoint:, :], [0])
            y_symmetry_constraint_value = (
                torch.norm(channel_top - channel_bottom, p=1) / pixel_total
            )
            y_symmetry_constraint += y_symmetry_constraint_value

        ce.c4 = y_symmetry_constraint

    # NOTE: I ran this with the inequality constraint for
    # single material and it seems that the inequality constraint
    # has a stronger preference than equality
    void_mass = (1 - x_phys[:, 0]).mean()
    void_constraint = (void_mass / args['volfrac']) - 1.0

    if num_materials > 2:
        ci = pygransoStruct()

        ci.c1 = void_constraint - 0.05
        ci.c2 = -void_constraint + 0.05

    else:
        ce.c5 = void_constraint

    print(
        compliance.item(),
        mass_constraint.item(),
        discrete_constraint.item(),
        x_symmetry_constraint.item(),
        y_symmetry_constraint.item(),
        void_constraint.item(),
        args['x_symmetry'],
        args['y_symmetry'],
    )

    # Save all calculations for analysis
    compliance_list.append(compliance.item())
    volume_constraint_list.append(mass_constraint.item())
    discrete_constraint_list.append(discrete_constraint.item())

    if args['x_symmetry']:
        x_symmetry_constraint_list.append(x_symmetry_constraint.item())

    if args['y_symmetry']:
        y_symmetry_constraint_list.append(y_symmetry_constraint.item())

    # Let's try and clear as much stuff as we can to preserve memory
    del x_phys, logits, mask, ke
    gc.collect()
    torch.cuda.empty_cache()

    return (f + f_void) / 2.0, ci, ce


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
    epsilon = 1e-3
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
    include_penalty=False,
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
    args,
    device,
    cnn_kwargs: Optional[dict] = None,
    *,
    num_trials: int = 20,
    mu: float = 1.0,
    maxit: int = 1500,
) -> dict[str, Any]:
    """
    Function to incorportate the multi-material paradigm for
    TO training
    """
    dtype = torch.double
    num_materials = len(args['e_materials']) + 1

    # Assign empty arrays to save during training
    trials_designs = np.zeros((num_trials, num_materials, args["nely"], args["nelx"]))
    trials_losses = np.full((maxit + 1, num_trials), np.nan)
    trials_volume_constraint = np.full((maxit + 1, num_trials), np.nan)
    trials_discrete_constraint = np.full((maxit + 1, num_trials), np.nan)
    trials_x_symmetry_constraint = np.full((maxit + 1, num_trials), np.nan)
    trials_y_symmetry_constraint = np.full((maxit + 1, num_trials), np.nan)

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args['young'],
        poisson=args['poisson'],
        device=device,
    ).double()

    for index, seed in enumerate(range(0, num_trials)):
        seed = (seed + 1) * 10
        # Set seeds for reproducibility
        models.set_seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        # Model
        if cnn_kwargs is not None:
            model = models.MultiMaterialCNNModel(
                args, random_seed=seed, **cnn_kwargs
            ).to(device=device, dtype=dtype)
        else:
            model = models.MultiMaterialCNNModel(args, random_seed=seed).to(
                device=device, dtype=dtype
            )

        # Calculate the initial compliance
        model.eval()
        with torch.no_grad():
            # Calculate initial compliance for multi-material
            # channels
            (
                initial_compliance,
                initial_x_phys,
                _,
                _,
            ) = topo_physics.calculate_multi_material_compliance(
                model,
                ke,
                args,
                device,
                dtype,
            )

            # Calculate compliance of the void channel
            (initial_void_compliance, _, _,) = topo_physics.calculate_void_compliance(
                model,
                ke,
                args,
                device,
                dtype,
            )

        # Detach the initial values so we can use them during
        # calculations as a scaling factor
        initial_compliance = (
            torch.ceil(initial_compliance.to(dtype=torch.float64).detach()) + 1.0
        )
        initial_void_compliance = (
            torch.ceil(initial_void_compliance.to(dtype=torch.float64).detach()) + 1.0
        )

        # Put the model back into training model
        model.train()

        # Combined function lists. We use these lists to save information
        # during training.
        compliance_list: List[float] = []
        volume_constraint_list: List[float] = []
        discrete_constraint_list: List[float] = []
        x_symmetry_constraint_list: List[float] = []
        y_symmetry_constraint_list: List[float] = []

        # initialize the pygranso combined function
        combined_fn = lambda model: multi_material_constraint_function_v2(  # noqa
            model,
            initial_compliance,
            initial_void_compliance,
            ke,
            args,
            compliance_list,  # noqa
            volume_constraint_list,  # noqa
            discrete_constraint_list,  # noqa
            x_symmetry_constraint_list,  # noqa
            y_symmetry_constraint_list,  # noqa
            device,
            dtype,
        )

        # Setup pygranso solver inputs
        opts = pygransoStruct()
        opts.torch_device = device

        # Get the parameters for the neural network
        nvar = getNvarTorch(model.parameters())
        parameters = torch.nn.utils.parameters_to_vector(model.parameters())
        opts.x0 = parameters.detach().reshape(nvar, 1).to(device=device, dtype=dtype)

        # Additional pygranso options
        opts.limited_mem_size = 20
        opts.double_precision = True
        opts.mu0 = mu
        opts.maxit = maxit
        opts.print_frequency = 1
        opts.stat_l2_model = False
        opts.viol_eq_tol = 1e-4
        opts.opt_tol = 1e-4

        # Setup logging for pygranso
        mHLF_obj = utils.HaltLog()
        halt_log_fn, get_log_fn = mHLF_obj.makeHaltLogFunctions(opts.maxit)

        #  Set PyGRANSO's logging function in opts
        opts.halt_log_fn = halt_log_fn

        # Main algorithm with logging enabled.
        soln = pygranso(var_spec=model, combined_fn=combined_fn, user_opts=opts)

        # Get iterative values
        log = get_log_fn()
        log_f = pd.Series(log.f) * initial_compliance.cpu().numpy()
        len_log_f = len(log_f)

        # Get indexes of iterative values from pygranso
        indexes = (pd.Series(log.fn_evals).cumsum() - 1).values.tolist()

        # Get the final design
        model.eval()
        with torch.no_grad():
            (
                _,
                _,
                final_design,  # Use the logits so we do not have to reshape
                _,
            ) = topo_physics.calculate_multi_material_compliance(
                model,
                ke,
                args,
                device,
                dtype,
            )

        final_design = final_design.detach().cpu().numpy()

        # Get the final compliance
        final_compliance = soln.final.f * initial_compliance.item()

        # Save all trials data
        trials_designs[index, :, :, :] = final_design

        compliance_array = np.asarray(compliance_list)
        trials_losses[:len_log_f, index] = compliance_array[indexes]

        volume_constraint_array = np.asarray(volume_constraint_list)
        trials_volume_constraint[:len_log_f, index] = volume_constraint_array[indexes]

        discrete_constraint_array = np.asarray(discrete_constraint_list)
        trials_discrete_constraint[:len_log_f, index] = discrete_constraint_array[
            indexes
        ]

        x_symmetry_constraint_array = np.zeros(1)
        if args['x_symmetry']:
            x_symmetry_constraint_array = np.asarray(x_symmetry_constraint_list)
            trials_x_symmetry_constraint[
                :len_log_f, index
            ] = x_symmetry_constraint_array[indexes]

        y_symmetry_constraint_array = np.zeros(1)
        if args["y_symmetry"]:
            y_symmetry_constraint_array = np.asarray(y_symmetry_constraint_list)
            trials_y_symmetry_constraint[
                :len_log_f, index
            ] = y_symmetry_constraint_array[indexes]

        # Delete all variables for the next trial
        del (
            model,
            combined_fn,
            opts,
            mHLF_obj,
            halt_log_fn,
            get_log_fn,
            soln,
            log,
            final_design,
            final_compliance,
            compliance_list,
            volume_constraint_array,
            volume_constraint_list,
            discrete_constraint_array,
            discrete_constraint_list,
            x_symmetry_constraint_array,
            x_symmetry_constraint_list,
            y_symmetry_constraint_array,
            y_symmetry_constraint_list,
        )
        gc.collect()
        torch.cuda.empty_cache()

    outputs = {
        "designs": trials_designs,
        "losses": trials_losses,
        "volumes": trials_volume_constraint,
        "binary_constraint": trials_discrete_constraint,
        "x_symmetry_constraint": trials_x_symmetry_constraint,
        "y_symmetry_constraint": trials_y_symmetry_constraint,
    }

    return outputs


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
