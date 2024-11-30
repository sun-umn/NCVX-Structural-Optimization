#!/usr/bin/python
# stdlib
import os
import pickle
import warnings
from typing import Tuple

# third party
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray
from matplotlib import colors
from neural_structural_optimization import models as google_models
from neural_structural_optimization import topo_api as google_topo_api
from neural_structural_optimization import train as google_train

# first party
import mmto.build_mmto as cmmto  # noqa
import models
import problems
import topo_api
import topo_physics
import train
import utils
import wandb
from MMTOuNN.neuralTO_MM import TopologyOptimizer as MMTO
from TOuNN.TOuNN import TopologyOptimizer

# Filter warnings
warnings.filterwarnings('ignore')

# Global variables
CNN_FEATURES = (256, 128, 64, 32, 16)
MODEL_CONFIGS = {
    'small': {
        'latent_size': 96,
        'dense_channels': 24,
        'conv_filters': tuple(feature // 4 for feature in CNN_FEATURES),
    },
    'medium': {
        'latent_size': 96,
        'dense_channels': 24,
        'conv_filters': tuple(feature // 3 for feature in CNN_FEATURES),
    },
    'large': {
        'latent_size': 96,
        'dense_channels': 24,
        'conv_filters': tuple(feature // 2 for feature in CNN_FEATURES),
    },
}

PROBLEM_CONFIGS = {
    # Medium Structures
    'mbb_beam_96x32_0.5': {
        'include_symmetry': False,
    },
    'anchored_suspended_bridge_128x128_0.1': {
        'include_symmetry': False,
    },
    'l_shape_0.4_128x128_0.3': {
        'include_symmetry': False,
    },
    'cantilever_beam_two_point_96x96_0.4': {
        'include_symmetry': True,
    },
    # Large Structures
    'mbb_beam_384x128_0.5': {
        'include_symmetry': False,
    },
    'anchored_suspended_bridge_192x192_0.1': {
        'include_symmetry': False,
    },
    'l_shape_0.4_192x192_0.3': {
        'include_symmetry': False,
    },
}


# Define the cli group
@click.group()
def cli():  # noqa
    pass


def calculate_mass_constraint(
    design, nelx, nely, material_density_weight, combined_frac
):
    """
    Compute the total mass constraint from a final design
    """
    design = np.array(design)

    material_density_weight = np.array(material_density_weight)
    total_mass = (
        np.max(material_density_weight)
        * design.shape[0]  # Expecting (nelx * nely, material) outputs
        * combined_frac
    )

    num_materials = len(material_density_weight)
    mass_constraint = np.zeros(num_materials)
    for index, density_weight in enumerate(material_density_weight):
        mass_constraint[index] = density_weight * np.sum(design[:, index + 1])

    return mass_constraint.sum() / total_mass - 1.0


def calculate_binary_constraint(design, mask, epsilon):
    """
    Function to compute the binary constraint
    """
    total_elements = len(design[mask].flatten())
    binary_constraint = (
        np.linalg.norm(design[mask] * (1 - design[mask]), ord=1) / total_elements
    ) - epsilon
    return np.round(binary_constraint, decimals=5)


def calculate_volume_constraint(design, mask, volume):
    """
    Function that computes the volume constraint
    """
    volume_constraint = (np.mean(design[mask]) / volume) - 1.0
    return np.round(volume_constraint, decimals=5)


def build_outputs(problem_name, outputs, mask, volume, requires_flip, epsilon=1e-3):
    """
    From each of the methods we will have an outputs
    based on the number of trials. This function
    will put together the outputs of the best trial.

    Outputs is a Dict object and should have keys:
    1. designs
    2. losses
    3. volumes
    4. binary_constraint
    5. trials_initial_volumnes
    """
    # Get the losses and sort from lowest to highest
    losses_df = pd.DataFrame(outputs["losses"])
    losses_df = losses_df.ffill()

    # Sort the final losses by index
    losses = np.min(losses_df, axis=0).values
    losses_indexes = np.argsort(losses)

    # Reorder outputs
    # losses
    losses_df = losses_df.iloc[:, losses_indexes]

    # final designs
    final_designs = outputs["designs"]
    final_designs = final_designs[losses_indexes, :, :]

    # Get all final objects
    best_final_design = final_designs[0, :, :]
    # Compute the binary and volume constraints
    binary_constraint = calculate_binary_constraint(
        design=best_final_design,
        mask=mask,
        epsilon=epsilon,
    )

    # volume constraint
    volume_constraint = calculate_volume_constraint(
        design=best_final_design,
        mask=mask,
        volume=volume,
    )

    if requires_flip:
        if (
            ("mbb" in problem_name)
            or ("l_shape" in problem_name)
            or ("cantilever" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design[:, ::-1], best_final_design]
            )

        elif (
            ("multistory" in problem_name)
            or ("thin" in problem_name)
            or ("michell" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design, best_final_design[:, ::-1]] * 2
            )

    # last row, first column (-1, 0)
    best_score = np.round(losses_df.iloc[-1, 0], 2)

    # Create metrics
    metrics = {
        'loss': outputs["losses"],
        'volume_constraint': outputs['volumes'],
        'binary_constraint': outputs['binary_constraint'],
        'symmetry_constraint': outputs['symmetry_constraint'],
    }

    return best_final_design, best_score, binary_constraint, volume_constraint, metrics


def build_google_outputs(
    problem_name, iterations, ds, mask, volume, requires_flip, epsilon=1e-3
):
    """
    Build the google outputs.
    TODO: I think I will want to extend this for multiple
    trials but will leave as a single trial for now
    """
    # Get the minimum loss for the benchmark methods
    losses = ds.loss.transpose().to_pandas().cummin().ffill()
    cnn_loss = np.round(losses["cnn-lbfgs"].min(), 2)
    mma_loss = np.round(losses["mma"].min(), 2)

    # Select the final step from the xarray
    final_designs = ds.design.sel(step=200, method="nearest").data

    # CNN final design
    cnn_final_design = final_designs[0, :, :]
    cnn_binary_constraint = calculate_binary_constraint(
        design=cnn_final_design, mask=mask, epsilon=epsilon
    )
    cnn_volume_constraint = calculate_volume_constraint(
        design=cnn_final_design,
        mask=mask,
        volume=volume,
    )

    # MMA final design
    mma_final_design = final_designs[1, :, :]
    mma_binary_constraint = calculate_binary_constraint(
        design=mma_final_design,
        mask=mask,
        epsilon=epsilon,
    )
    mma_volume_constraint = calculate_volume_constraint(
        design=mma_final_design,
        mask=mask,
        volume=volume,
    )

    if requires_flip:
        if (
            ("mbb" in problem_name)
            or ("l_shape" in problem_name)
            or ("cantilever" in problem_name)
        ):
            cnn_final_design = np.hstack([cnn_final_design[:, ::-1], cnn_final_design])
            mma_final_design = np.hstack([mma_final_design[:, ::-1], mma_final_design])

        if (
            ("multistory" in problem_name)
            or ("thin" in problem_name)
            or ("michell" in problem_name)
        ):
            cnn_final_design = np.hstack(
                [cnn_final_design, cnn_final_design[:, ::-1]] * 2
            )
            mma_final_design = np.hstack(
                [mma_final_design, mma_final_design[:, ::-1]] * 2
            )

    # Here compute the trajectories
    cnn_volume_constraint_trajectory = []
    cnn_binary_constraint_trajectory = []
    mma_volume_constraint_trajectory = []
    mma_binary_constraint_trajectory = []
    for i in range(iterations):
        # Get the intermediate designs
        design = ds.design.sel(step=i, method="nearest").data

        # CNN final design
        cnn_design = design[0, :, :]
        cnn_binary_constraint = calculate_binary_constraint(
            design=cnn_design, mask=mask, epsilon=epsilon
        )
        cnn_volume_constraint = calculate_volume_constraint(
            design=cnn_design,
            mask=mask,
            volume=volume,
        )
        cnn_volume_constraint_trajectory.append(cnn_volume_constraint)
        cnn_binary_constraint_trajectory.append(cnn_binary_constraint)

        # MMA final design
        mma_design = design[1, :, :]
        mma_binary_constraint = calculate_binary_constraint(
            design=mma_design,
            mask=mask,
            epsilon=epsilon,
        )
        mma_volume_constraint = calculate_volume_constraint(
            design=mma_design,
            mask=mask,
            volume=volume,
        )
        mma_volume_constraint_trajectory.append(mma_volume_constraint)
        mma_binary_constraint_trajectory.append(mma_binary_constraint)

    # Create the trajectories
    cnn_metrics = {
        'loss': losses["cnn-lbfgs"],
        'volume_constraint': cnn_volume_constraint_trajectory,
        'binary_constraint': cnn_binary_constraint_trajectory,
    }

    mma_metrics = {
        'loss': losses["mma"],
        'volume_constraint': mma_volume_constraint_trajectory,
        'binary_constraint': mma_binary_constraint_trajectory,
    }

    return {
        "google-cnn": (
            cnn_final_design,
            cnn_loss,
            cnn_binary_constraint,
            cnn_volume_constraint,
            cnn_metrics,
        ),
        "mma": (
            mma_final_design,
            mma_loss,
            mma_binary_constraint,
            mma_volume_constraint,
            mma_metrics,
        ),
    }


def train_all(problem, max_iterations, cnn_kwargs=None):
    """
    Function that will compute the MMA and google cnn
    structure optimization.
    """
    args = google_topo_api.specified_task(problem)
    if cnn_kwargs is None:
        cnn_kwargs = {}

    model = google_models.PixelModel(args=args)
    ds_mma = google_train.method_of_moving_asymptotes(model, max_iterations)

    model = google_models.CNNModel(args=args, **cnn_kwargs)
    ds_cnn = google_train.train_lbfgs(model, max_iterations)

    dims = pd.Index(["cnn-lbfgs", "mma"], name="model")

    return xarray.concat([ds_cnn, ds_mma], dim=dims)


def tounn_train_and_outputs(problem, requires_flip):
    """
    Function that will run the TOuNN pipeline
    """
    # Try setting seed here as well
    models.set_seed(0)

    # Get the problem name
    problem_name = problem.name

    # Get the arguments for the problem
    args = topo_api.specified_task(problem)

    # Get the problem dimensions
    # Our arguments come as torch tensors so we need to convert back
    # to numpy and int to work with their framework
    nelx, nely = int(args['nelx'].numpy()), int(args['nely'].numpy())

    # Get the volume fraction
    desiredVolumeFraction = args['volfrac']

    # Get the forces - convert to numpy to work with
    # their pipeline
    force = args['forces'].cpu().numpy()

    # Add an extra axis because their framework expects
    # (n, 1)
    force = force[:, None]

    # Get the fixed dofs
    fixed = args['fixdofs'].cpu().numpy()

    # Get epsilon value
    epsilon = args['epsilon']

    # Get the mask for tounn problems
    tounn_mask = args['tounn_mask']

    # TODO: Figure out how this works with non-design
    # regions but for now it will be none
    nonDesignRegion = {
        'Rect': tounn_mask,
        'Circ': None,
        'Annular': None,
    }

    # Symmetry about axes
    symXAxis = False
    symYAxis = False

    # Penal in their code starts at 2
    penal = 2

    # Neural network config
    numLayers = 5
    numNeuronsPerLyr = 20
    minEpochs = 20
    maxEpochs = 1500
    useSavedNet = False

    # Run the pipeline
    topOpt = TopologyOptimizer()

    # Initialize the FE (Finite element) solver
    topOpt.initializeFE(
        problem.name, nelx, nely, force, fixed, penal, nonDesignRegion, args=args
    )

    # Initialize the optimizer
    topOpt.initializeOptimizer(
        numLayers, numNeuronsPerLyr, desiredVolumeFraction, symXAxis, symYAxis
    )

    # Run the optimization
    topOpt.optimizeDesign(maxEpochs, minEpochs, useSavedNet)

    # After everything is fitted we need to extract the final information
    # Set the plotResolution to 1
    plotResolution = 1

    # compute the points for the problem
    xyPlot, nonDesignPlotIdx = topOpt.generatePoints(
        topOpt.FE.nelx, topOpt.FE.nely, plotResolution, topOpt.nonDesignRegion
    )

    # Compute the final density
    density = torch.flatten(topOpt.topNet(xyPlot, nonDesignPlotIdx))
    density = density.detach().cpu().numpy()
    best_final_design = density.copy()

    # The final design needs to be reshaped and transposed
    best_final_design = best_final_design.reshape(
        plotResolution * topOpt.FE.nelx, plotResolution * topOpt.FE.nely
    )
    best_final_design = best_final_design.T

    # get the best score
    best_score = np.round(topOpt.convergenceHistory[-1][4], 2)

    # Return everything
    mask = (torch.broadcast_to(args["mask"], (nely, nelx)) > 0).cpu().numpy()

    # Compute the binary constraint
    binary_constraint = calculate_binary_constraint(
        design=best_final_design,
        mask=mask,
        epsilon=epsilon,
    )
    volume_constraint = calculate_volume_constraint(
        design=best_final_design,
        mask=mask,
        volume=desiredVolumeFraction,
    )

    # Add more information about the outputs
    if requires_flip:
        if (
            ("mbb" in problem_name)
            or ("l_shape" in problem_name)
            or ("cantilever" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design[:, ::-1], best_final_design]
            )

        if (
            ("multistory" in problem_name)
            or ("thin" in problem_name)
            or ("michell" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design, best_final_design[:, ::-1]] * 2
            )

    # Here will create a dict to save the losses
    metrics = {
        'loss': topOpt.convergenceHistory[4],
        'volume_constraint': topOpt.convergenceHistory[5],
        'binary_constraint': topOpt.convergenceHistory[6],
    }
    metrics = {
        'loss': np.array(
            [value for _, _, _, _, value, _, _ in topOpt.convergenceHistory]
        ),
        'volume_constraint': np.array(
            [value for _, _, _, _, _, value, _ in topOpt.convergenceHistory]
        ),
        'binary_constraint': np.array(
            [value for _, _, _, _, _, _, value in topOpt.convergenceHistory]
        ),
    }

    return best_final_design, best_score, binary_constraint, volume_constraint, metrics


def mmtounn_train_and_outputs(
    args, nelx, nely, e_materials, material_density_weight, combined_frac, seed=1234
):
    """
    Function that will run the TOuNN pipeline
    """
    # Set a different seed?

    elemArea = 1.0

    # Network config
    numLayers = 5
    # the depth of the NN
    numNeuronsPerLyr = 20
    # the height of the NN

    # problem
    exampleName = 'TipCantilever'

    fixed = args['fixdofs'].numpy().astype(int)
    force = args['forces'].numpy().astype(np.float64)
    force = force[..., np.newaxis]

    # e_materials and material_density_weight need to be numpy arrays
    e_materials = e_materials.numpy()
    material_density_weight = material_density_weight.numpy()

    nonDesignRegion = None
    symXAxis = False
    symYAxis = False

    # Additional config
    minEpochs = 50
    maxEpochs = 1000
    penal = 1.0
    useSavedNet = False
    device = 'cpu'

    # Compute the topology
    topOpt = MMTO()
    topOpt.initializeFE(
        exampleName,
        nelx,
        nely,
        elemArea,
        force,
        fixed,
        device,
        penal,
        nonDesignRegion,
        e_materials,
    )
    topOpt.initializeOptimizer(
        numLayers=numLayers,
        numNeuronsPerLyr=numNeuronsPerLyr,
        desiredMassFraction=combined_frac,
        massDensityMaterials=material_density_weight,
        symXAxis=symXAxis,
        symYAxis=symYAxis,
        seed=seed,
    )

    # Run the optimization
    _ = topOpt.train(maxEpochs, minEpochs, useSavedNet)

    # Get the density outputs
    plotResolution = 1
    xyPlot, nonDesignPlotIdx = topOpt.generatePoints(
        topOpt.FE.nelx, topOpt.FE.nely, plotResolution, topOpt.nonDesignRegion
    )

    # Compute the final density
    density = topOpt.topNet(xyPlot, nonDesignPlotIdx)
    density = density.detach().cpu().numpy()

    return topOpt, density


def run_classical_mmto(
    args, nelx, nely, ke, x0, volfrac, costfrac, penal, rmin, D, E, P, MinMove
) -> Tuple[np.ndarray, float, float]:
    """
    Function to run classical MMTO
    """
    # Initialize a grid
    x = np.ones((nely, nelx)) * x0

    # The incoming variables will need to be converted to
    # numpy arrays
    D = np.array(D)
    E = np.array(E)
    P = np.array(P)

    # Make sure arguments important for args are also
    # converted
    ke = np.array(ke)
    args['forces'] = np.array(args['forces'])
    args['fixdofs'] = np.array(args['fixdofs'])
    args['freedofs'] = np.array(args['freedofs'])

    # We will loop until converged
    loop = 0
    change = 1.0

    while change > 1.01 * MinMove:
        loop = loop + 1
        xold = x

        # Run through the interpolation and FE
        E_, dE_ = cmmto.ordered_simp_interpolation(
            nelx=nelx,
            nely=nely,
            x=x,
            penal=penal,
            X=D,
            Y=E,
        )
        P_, dP_ = cmmto.ordered_simp_interpolation(
            nelx=nelx,
            nely=nely,
            x=x,
            penal=(1 / penal),
            X=D,
            Y=P,
        )

        # Get the displacement vector
        U = cmmto.finite_element(
            args=args,
            nelx=nelx,
            nely=nely,
            E_Interpolation=E_,
            KE=ke,
        )

        # Objective function and sensitivity analysis
        c = 0.0
        dc = np.zeros((nely, nelx))

        for ely in range(nely):
            for elx in range(nelx):
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely

                edof = np.array(
                    [
                        2 * n1 - 1,
                        2 * n1,
                        2 * n2 - 1,
                        2 * n2,
                        2 * n2 + 1,
                        2 * n2 + 2,
                        2 * n1 + 1,
                        2 * n1 + 2,
                    ]
                )
                edof = edof + 1

                Ue = U[edof, 0]

                c += E_[ely, elx] * np.dot(Ue.T, np.dot(ke, Ue))
                dc[ely, elx] = -dE_[ely, elx] * np.dot(Ue.T, np.dot(ke, Ue))

        # Filtering of sensitivities
        dc = cmmto.check(nelx=nelx, nely=nely, rmin=rmin, x=x, dc=dc)

        # Update the design with the optimality criterion
        x = cmmto.optimality_criterion(
            nelx=nelx,
            nely=nely,
            x=x,
            volfrac=volfrac,
            costfrac=costfrac,
            dc=dc,
            E_=E_,
            dE_=dE_,
            P_=P_,
            dP_=dP_,
            loop=loop,
            MinMove=MinMove,
        )

        change = np.max(np.abs(x - xold))
        mass_fraction = np.sum(x) / (nelx * nely)

        print(f'Iteration = {loop}; Compliance = {c}; Mass Fraction = {mass_fraction}')

        # Early stopping
        if loop >= 500:
            break

    return x, c, mass_fraction


@cli.command('run-multi-material-pipeline')
@click.option('--problem_name', default='tip_cantilever_beam')
def run_multi_material_pipeline(problem_name):
    """
    Function to run the multi-material pipeline
    """
    print(f'Problem Name = {problem_name}')
    device = torch.device('cpu')
    first_stage_maxit = 5
    # second_stage_maxit = 500

    # For testing we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    # TODO: Save training and validation curves per fold
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=['ntopco-mmto-task'],
    )

    # Will create directories for saving models
    save_path = os.path.join(
        '/home/jusun/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    # Problem specifications
    if problem_name == 'tip_cantilever_beam':
        nelx = 64
        nely = 32
        combined_frac = 0.6
        e_materials = torch.tensor([0.0, 3.0, 2.0, 1.0], dtype=torch.double)
        material_density_weight = torch.tensor([0.0, 1.0, 0.7, 0.4])
        P = torch.tensor([1.0, 1.0, 1.0, 1.0])
        costfrac = 1.0

        args = topo_api.multi_material_tip_cantilever_task(
            nelx=nelx,
            nely=nely,
            e_materials=e_materials,
            material_density_weight=material_density_weight,
            combined_frac=combined_frac,
        )

    elif problem_name == 'bridge':
        nelx = 128
        nely = 64
        combined_frac = 0.4
        e_materials = torch.tensor([0.0, 0.2, 0.6, 1.0], dtype=torch.double)
        material_density_weight = torch.tensor([0.0, 0.4, 0.7, 1.0])
        P = torch.tensor([1.0, 1.0, 1.0, 1.0])
        costfrac = 1.0

        args = topo_api.multi_material_bridge_task(
            nelx=nelx,
            nely=nely,
            e_materials=e_materials,
            material_density_weight=material_density_weight,
            combined_frac=combined_frac,
        )

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args['young'],
        poisson=args['poisson'],
        device=device,
    ).double()

    # When running the classical method it is deterministic so we do not need
    # multiple runs
    cmmto_x_phys, cmmto_compliance, cmmto_mass_constraint = run_classical_mmto(
        args=args,
        nelx=nelx,
        nely=nely,
        ke=ke,
        x0=0.5,
        volfrac=combined_frac,
        costfrac=costfrac,
        penal=3.0,
        rmin=2.5,
        D=material_density_weight,
        E=e_materials,
        P=P,
        MinMove=0.001,
    )

    cmmto_outputs = {
        'final_design': cmmto_x_phys,
        'compliance': cmmto_compliance,
        'mass_constraint': cmmto_mass_constraint - combined_frac,
        'material_density_weight': material_density_weight,
        'nelx': nelx,
        'nely': nely,
    }

    cmmto_filepath = os.path.join(save_path, 'cmmto.pickle')
    with open(cmmto_filepath, 'wb') as handle:
        pickle.dump(cmmto_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Setup for our method
    args['penal'] = 1.0
    args['forces'] = torch.tensor(args['forces'].ravel())
    args['fixdofs'] = torch.tensor(args['fixdofs'])
    args['freedofs'] = torch.tensor(args['freedofs'])

    # After running Classical method we need to update e_materials
    # and material_density_weight
    e_materials = e_materials[1:]
    material_density_weight = material_density_weight[1:]
    args['e_materials'] = e_materials
    args['material_density_weight'] = material_density_weight

    # DIP Setup
    conv_filters = (256, 128, 64, 32)
    cnn_kwargs = {
        'latent_size': 128,
        'dense_channels': 32,
        'kernel_size': (5, 5),
        'conv_filters': conv_filters,
    }

    # Trials and seeds
    seeds = [1234, 1985]
    for seed in seeds:
        # Intialize random seed
        utils.build_random_seed(seed)
        cnn_kwargs['random_seed'] = seed

        model = models.MultiMaterialCNNModel(args, **cnn_kwargs).to(
            device=device, dtype=torch.double
        )

        # Calculate the initial compliance
        model.eval()
        with torch.no_grad():
            (
                initial_compliance,
                x_phys,
                _,
            ) = topo_physics.calculate_multi_material_compliance(
                model, ke, args, device, torch.double
            )

        # Detach calculation and use it for scaling in PyGranso
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0
        )

        # Train PyGranso MMTO - First Stage
        # Setup the combined function for PyGranso
        comb_fn = lambda model: train.multi_material_constraint_function(  # noqa
            model,
            initial_compliance,
            ke,
            args,
            add_constraints=True,
            device=device,
            dtype=torch.double,
        )

        train.train_pygranso_mmto(
            model=model, comb_fn=comb_fn, maxit=first_stage_maxit, device=device
        )

        # Get the final design
        compliance, final_design, _ = topo_physics.calculate_multi_material_compliance(
            model, ke, args, device, torch.double
        )
        final_design = final_design.detach().numpy()

        # Compute mass constraint
        ntopco_mass_constraint = calculate_mass_constraint(
            design=final_design,
            nelx=nelx,
            nely=nely,
            material_density_weight=material_density_weight,
            combined_frac=combined_frac,
        )

        # TODO: Extract all of the relevant information
        ntopco_outputs = {
            'final_design': final_design,
            'compliance': compliance,
            'mass_constraint': ntopco_mass_constraint,
            'material_density_weight': material_density_weight,
            'nelx': nelx,
            'nely': nely,
            'cnn_kwargs': cnn_kwargs,
        }

        # Compute the final design and save to experiments
        ntopco_filepath = os.path.join(save_path, f'ntopco-{seed}.pickle')
        with open(ntopco_filepath, 'wb') as handle:
            pickle.dump(ntopco_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Run MM-TOuNN Pipeline
        topOpt, mmtounn_final_design = mmtounn_train_and_outputs(
            args=args,
            nelx=nelx,
            nely=nely,
            e_materials=e_materials,
            material_density_weight=material_density_weight,
            combined_frac=combined_frac,
            seed=seed,
        )
        print('MM-TOuNN Completed! 🎉')

        # Final compliance
        mmtounn_compliance = topOpt.convergenceHistory[-1][-1]

        # Compute mass constraint
        mmtounn_mass_constraint = calculate_mass_constraint(
            design=mmtounn_final_design,
            nelx=nelx,
            nely=nely,
            material_density_weight=material_density_weight,
            combined_frac=combined_frac,
        )

        mmtounn_outputs = {
            'final_design': mmtounn_final_design,
            'compliance': mmtounn_compliance,
            'mass_constraint': mmtounn_mass_constraint,
            'material_density_weight': material_density_weight,
            'nelx': nelx,
            'nely': nely,
        }

        # Compute the final design and save to experiments
        mmtounn_filepath = os.path.join(save_path, f'mmtounn-{seed}.pickle')
        with open(mmtounn_filepath, 'wb') as handle:
            pickle.dump(mmtounn_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Multi-Material Pipeline Completed! 🏆')


@cli.command('run-multi-structure-pipeline-v2')
@click.option('--model_size', default='medium')
@click.option('--problem_name', default='mbb_beam_96x32_0.5')
@click.option('--kernel_size', default="12,12")
@click.option('--num_trials', default=1)
def run_multi_structure_pipeline_v2(model_size, problem_name, kernel_size, num_trials):
    """
    Task that will build out multiple structures and compare
    performance against known benchmarks.
    """
    # Top level config
    device = utils.get_devices()

    # Max iterations for PyGranso
    maxit = 1

    # Max iterations for Google-DIP
    max_iterations = 2

    # For testing, we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    # Enable wandb
    wandb.login(key=API_KEY)

    # CNN kwargs
    kernel_size_tuple = tuple(int(i) for i in kernel_size.split(','))
    cnn_kwargs = MODEL_CONFIGS[model_size]
    cnn_kwargs['kernel_size'] = kernel_size_tuple

    # Initalize wandb
    trial_tag = 'single'
    if num_trials > 1:
        trial_tag = 'multi-trial'
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=[
            'topology-optimization-task',
            model_size,
            problem_name,
            trial_tag,
        ],
        config={
            'latent_size': cnn_kwargs['latent_size'],
            'dense_channels': cnn_kwargs['dense_channels'],
            'conv_filters': cnn_kwargs['conv_filters'],
            'kernel_size': kernel_size,
        },
    )

    # Create directory for saving the model and
    # output data
    save_path = os.path.join(
        '/home/jusun/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    # Write the configuration
    config_filepath = os.path.join(save_path, 'config.txt')
    with open(config_filepath, 'w') as f:
        f.write(
            f'model size = {model_size}; structure = {problem_name};'
            f'cnn_kwargs = {cnn_kwargs}'
        )

    # PyGranso function
    comb_fn = train.volume_constrained_structural_optimization_function

    # Build the problems for pygranso and google
    PYGRANSO_PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)  # noqa
    PROBLEM_CONFIG = PROBLEM_CONFIGS[problem_name]  # noqa
    include_symmetry = PROBLEM_CONFIG['include_symmetry']

    # Old configs that need to be refactored
    requires_flip = False
    total_frames = 1

    # Build structure
    print(f"Building structure: {problem_name}")
    problem = PYGRANSO_PROBLEMS_BY_NAME.get(problem_name)

    # Get volume assignment
    args = topo_api.specified_task(problem, device=device)
    volume = args["volfrac"]

    nely = int(args["nely"])
    nelx = int(args["nelx"])
    mask = (torch.broadcast_to(args["mask"], (nely, nelx)) > 0).cpu().numpy()

    # Build the structure with pygranso
    outputs = train.train_pygranso(
        problem=problem,
        device=device,
        pygranso_combined_function=comb_fn,
        requires_flip=requires_flip,
        total_frames=total_frames,
        cnn_kwargs=cnn_kwargs,
        num_trials=num_trials,
        maxit=maxit,
        include_symmetry=include_symmetry,
    )

    # Build the outputs
    # NTO-PCO
    pygranso_outputs = build_outputs(
        problem_name=problem_name,
        outputs=outputs,
        mask=mask,
        volume=volume,
        requires_flip=requires_flip,
    )

    # TOuNN Outputs
    tounn_outputs = tounn_train_and_outputs(problem, requires_flip)

    # Build Google-DIP results - We will use our problem library
    # so we can also have custom structures not in the
    # google code
    google_problem = problem

    # Set to numpy for google framework
    google_problem.normals = google_problem.normals.cpu().numpy()
    google_problem.forces = google_problem.forces.cpu().numpy()

    if not isinstance(google_problem.mask, int):
        google_problem.mask = google_problem.mask.cpu().numpy()

    # Google specific parameters
    google_problem.mirror_left = True
    google_problem.mirror_right = False

    # Same interface and usage as Google-DIP code
    ds = train_all(google_problem, max_iterations)

    # Google-DIP outputs
    benchmark_outputs = build_google_outputs(
        problem_name=problem_name,
        iterations=max_iterations,
        ds=ds,
        mask=mask,
        volume=volume,
        requires_flip=requires_flip,
    )

    # Save all outputs
    # NTO-PCO Outputs
    pygranso_filepath = os.path.join(save_path, f'{problem_name}-pygranso-cnn.pickle')
    with open(pygranso_filepath, 'wb') as handle:
        pickle.dump(pygranso_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TOuNN Outputs
    tounn_filepath = os.path.join(save_path, f'{problem_name}-tounn.pickle')
    with open(tounn_filepath, 'wb') as handle:
        pickle.dump(tounn_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Google-DIP Outputs
    google_filepath = os.path.join(save_path, f'{problem_name}-google.pickle')
    with open(google_filepath, 'wb') as handle:
        pickle.dump(benchmark_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Run completed! 🎉')


@cli.command('test-multi-material-bridge')
def test_multi_material_bridge():
    # For testing, we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    penal = 3.0
    scale_factor = 1.0
    latent_size = 96
    dense_channels = 24
    kernel_size = (11, 11)
    ks_string = ",".join(map(str, kernel_size))

    # DIP Setup
    conv_filters = (256 // 2, 128 // 2, 64 // 2, 32 // 2)
    cv_string = ",".join(map(str, conv_filters))

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=[
            'topology-optimization-task',
            'multi-material bridge',
        ],
        config={
            'latent_size': latent_size,
            'dense_channels': dense_channels,
            'kernel_size': ks_string,
            'conv_filters': cv_string,
            'penal': penal,
            'scale_factor': scale_factor,
        },
    )

    # Create directory for saving the model and
    # output data
    save_path = os.path.join(
        '/users/5/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    device = torch.device('cpu')

    # Setup the multi-material bridge problem
    nelx = 128
    nely = 64
    combined_frac = 0.4
    e_materials = torch.tensor([0.2, 0.6, 1.0], dtype=torch.double)
    material_density_weight = torch.tensor([0.4, 0.7, 1.0])

    args = topo_api.multi_material_bridge_task(
        nelx=nelx,
        nely=nely,
        e_materials=e_materials,
        material_density_weight=material_density_weight,
        combined_frac=combined_frac,
    )

    args['penal'] = penal
    args['scale_factor'] = scale_factor
    args['forces'] = torch.tensor(args['forces'].ravel())
    args['fixdofs'] = torch.tensor(args['fixdofs'])
    args['freedofs'] = torch.tensor(args['freedofs'])
    args['e_materials'] = e_materials
    args['material_density_weight'] = material_density_weight

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args['young'],
        poisson=args['poisson'],
        device=device,
    ).double()

    cnn_kwargs = {
        'latent_size': latent_size,
        'dense_channels': dense_channels,
        'kernel_size': kernel_size,
        'conv_filters': conv_filters,
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 6))
    axes = axes.flatten()

    # Trials and seeds
    seeds = [1234, 1985, 1986, 2009]
    first_stage_maxit = 1500
    for idx, seed in enumerate(seeds):
        ax = axes[idx]

        # Intialize random seed
        utils.build_random_seed(seed)
        cnn_kwargs['random_seed'] = seed

        model = models.MultiMaterialCNNModel(args, **cnn_kwargs).to(
            device=device, dtype=torch.double
        )

        # Calculate the initial compliance
        model.eval()
        with torch.no_grad():
            (
                initial_compliance,
                x_phys,
                _,
                _,
            ) = topo_physics.calculate_multi_material_compliance(
                model, ke, args, device, torch.double
            )

        # Detach calculation and use it for scaling in PyGranso
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0
        )

        # Train PyGranso MMTO - First Stage
        # Setup the combined function for PyGranso
        comb_fn = lambda model: train.multi_material_constraint_function(  # noqa
            model,
            initial_compliance,
            ke,
            args,
            device=device,
            dtype=torch.double,
        )

        train.train_pygranso_mmto(
            model=model, comb_fn=comb_fn, maxit=first_stage_maxit, device=device
        )

        # Get the final design
        (
            compliance,
            final_design,
            _,
            _,
        ) = topo_physics.calculate_multi_material_compliance(
            model, ke, args, device, torch.double
        )
        final_design = final_design.detach().numpy()

        design = final_design.argmax(axis=1).reshape(128, 64).T

        fillColors = ['white', 'black', 'red', 'blue']
        cmap = colors.ListedColormap(fillColors)

        ax.imshow(design, cmap=cmap, aspect="auto")
        ax.set_title(f'Compliance = {compliance}')
        ax.set_xticks([])
        ax.set_yticks([])

    path = os.path.join(save_path, 'multi-bridge-results.png')
    fig.savefig(path)


@cli.command('test-multi-material-cantilever')
def test_multi_material_cantilever():
    # For testing, we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    penal = 3.0
    scale_factor = 1.0
    latent_size = 96
    dense_channels = 24
    kernel_size = (11, 11)
    ks_string = ",".join(map(str, kernel_size))

    # DIP Setup
    conv_filters = (256 // 2, 128 // 2, 64 // 2, 32 // 2)
    cv_string = ",".join(map(str, conv_filters))

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=[
            'topology-optimization-task',
            'multi-material cantilever',
        ],
        config={
            'latent_size': latent_size,
            'dense_channels': dense_channels,
            'kernel_size': ks_string,
            'conv_filters': cv_string,
            'penal': penal,
            'scale_factor': scale_factor,
        },
    )

    # Create directory for saving the model and
    # output data
    save_path = os.path.join(
        '/users/5/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    device = torch.device('cpu')

    # Setup the multi-material bridge problem
    nelx = 64
    nely = 32
    combined_frac = 0.6
    e_materials = torch.tensor([3.0, 2.0, 1.0], dtype=torch.double)
    material_density_weight = torch.tensor([1.0, 0.7, 0.4])

    args = topo_api.multi_material_tip_cantilever_task(
        nelx=nelx,
        nely=nely,
        e_materials=e_materials,
        material_density_weight=material_density_weight,
        combined_frac=combined_frac,
    )

    args['penal'] = penal
    args['young_min'] = 1e-9
    args['scale_factor'] = scale_factor
    args['forces'] = torch.tensor(args['forces'].ravel())
    args['fixdofs'] = torch.tensor(args['fixdofs'])
    args['freedofs'] = torch.tensor(args['freedofs'])
    args['e_materials'] = e_materials
    args['material_density_weight'] = material_density_weight

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args['young'],
        poisson=args['poisson'],
        device=device,
    ).double()

    cnn_kwargs = {
        'latent_size': latent_size,
        'dense_channels': dense_channels,
        'kernel_size': kernel_size,
        'conv_filters': conv_filters,
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 6))
    axes = axes.flatten()

    # Trials and seeds
    seeds = [1234, 1985, 1986, 2009]
    first_stage_maxit = 1500
    for idx, seed in enumerate(seeds):
        try:
            ax = axes[idx]
            # Intialize random seed
            utils.build_random_seed(seed)
            cnn_kwargs['random_seed'] = seed

            model = models.MultiMaterialCNNModel(args, **cnn_kwargs).to(
                device=device, dtype=torch.double
            )

            # Calculate the initial compliance
            model.eval()
            with torch.no_grad():
                (
                    initial_compliance,
                    x_phys,
                    _,
                    _,
                ) = topo_physics.calculate_multi_material_compliance(
                    model, ke, args, device, torch.double
                )

            # Detach calculation and use it for scaling in PyGranso
            initial_compliance = (
                torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0
            )

            # Train PyGranso MMTO - First Stage
            # Setup the combined function for PyGranso
            comb_fn = lambda model: train.multi_material_constraint_function(  # noqa
                model,
                initial_compliance,
                ke,
                args,
                device=device,
                dtype=torch.double,
            )

            train.train_pygranso_mmto(
                model=model, comb_fn=comb_fn, maxit=first_stage_maxit, device=device
            )

            # Get the final design
            (
                compliance,
                final_design,
                _,
                _,
            ) = topo_physics.calculate_multi_material_compliance(
                model, ke, args, device, torch.double
            )
            final_design = final_design.detach().numpy()

            design = final_design.argmax(axis=1).reshape(64, 32).T[::-1, :]

            fillColors = ['white', 'black', 'red', 'blue']
            cmap = colors.ListedColormap(fillColors)

            ax.imshow(design, cmap=cmap, aspect="auto")
            ax.set_title(f'Compliance = {compliance}')
            ax.set_xticks([])
            ax.set_yticks([])

        except:
            pass

    path = os.path.join(save_path, 'multi-cantilever-results.png')
    fig.savefig(path)


@cli.command('test-single-material-bridge-v2')
def test_single_material_bridge_v2():
    # For testing, we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    penal = 3.0
    scale_factor = 1.0
    latent_size = 96
    dense_channels = 24
    kernel_size = (11, 11)
    ks_string = ",".join(map(str, kernel_size))

    conv_filters = (256 // 2, 128 // 2, 64 // 2, 32 // 2)
    cv_string = ",".join(map(str, conv_filters))

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=[
            'topology-optimization-task',
            'single-material bridge',
        ],
        config={
            'latent_size': latent_size,
            'dense_channels': dense_channels,
            'conv_filters': cv_string,
            'kernel_size': ks_string,
            'penal': penal,
            'scale_factor': scale_factor,
        },
    )

    # Create directory for saving the model and
    # output data
    save_path = os.path.join(
        '/users/5/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    device = torch.device('cpu')

    problem_name = 'anchored_suspended_bridge_128x128_0.1'

    # Build the problems for pygranso and google
    PYGRANSO_PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)  # noqa

    # Build structure
    print(f"Building structure: {problem_name}")
    problem = PYGRANSO_PROBLEMS_BY_NAME.get(problem_name)

    args = topo_api.specified_task(problem, device=device)
    args['penal'] = penal
    args['scale_factor'] = scale_factor
    args['young_min'] = 1e-9
    args['y_symmetry'] = False
    args['x_symmetry'] = False

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args['young'],
        poisson=args['poisson'],
        device=device,
    ).double()

    cnn_kwargs = {
        'latent_size': latent_size,
        'dense_channels': dense_channels,
        'kernel_size': kernel_size,
        'conv_filters': conv_filters,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes = axes.flatten()

    first_stage_maxit = 4000

    # Trials and seeds
    seeds = [0, 10]
    for idx, seed in enumerate(seeds):
        ax = axes[idx]

        # Intialize random seed
        utils.build_random_seed(seed)
        cnn_kwargs['random_seed'] = seed

        model = models.MultiMaterialCNNModel(args, **cnn_kwargs).to(
            device=device, dtype=torch.double
        )

        # Calculate the initial compliance
        model.eval()
        with torch.no_grad():
            (
                initial_compliance,
                x_phys,
                _,
                _,
            ) = topo_physics.calculate_multi_material_compliance(
                model, ke, args, device, torch.double
            )

        # Detach calculation and use it for scaling in PyGranso
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0
        )

        # Train PyGranso MMTO - First Stage
        # Setup the combined function for PyGranso
        comb_fn = lambda model: train.multi_material_constraint_function(  # noqa
            model,
            initial_compliance,
            ke,
            args,
            device=device,
            dtype=torch.double,
        )

        train.train_pygranso_mmto(
            model=model, comb_fn=comb_fn, maxit=first_stage_maxit, device=device
        )

        # Get the final design
        (
            compliance,
            final_design,
            _,
            _,
        ) = topo_physics.calculate_multi_material_compliance(
            model, ke, args, device, torch.double
        )
        final_design = final_design.detach().numpy()

        design = final_design[:, 1].reshape(128, 128).T

        ax.imshow(design, cmap='Greys', aspect="auto")
        ax.set_title(f'Compliance = {compliance}')
        ax.set_xticks([])
        ax.set_yticks([])

    path = os.path.join(save_path, 'single-bridge-results.png')
    fig.savefig(path)


@cli.command('test-single-material-l-shape')
def test_single_material_l_shape():
    # For testing, we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    penal = 3.0
    scale_factor = 1.0
    latent_size = 96
    dense_channels = 24
    kernel_size = (11, 11)
    ks_string = ",".join(map(str, kernel_size))

    # DIP Setup
    conv_filters = (256 // 2, 128 // 2, 64 // 2, 32 // 2)
    cv_string = ",".join(map(str, conv_filters))

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=[
            'topology-optimization-task',
            'single-material l-shape',
        ],
        config={
            'latent_size': latent_size,
            'dense_channels': dense_channels,
            'kernel_size': ks_string,
            'conv_filters': cv_string,
            'penal': penal,
            'scale_factor': scale_factor,
        },
    )

    # Create directory for saving the model and
    # output data
    save_path = os.path.join(
        '/users/5/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    device = torch.device('cpu')

    problem_name = 'l_shape_0.4_128x128_0.3'

    # Build the problems for pygranso and google
    PYGRANSO_PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)  # noqa

    # Build structure
    print(f"Building structure: {problem_name}")
    problem = PYGRANSO_PROBLEMS_BY_NAME.get(problem_name)

    args = topo_api.specified_task(problem, device=device)
    args['penal'] = penal
    args['scale_factor'] = scale_factor
    args['young_min'] = 1e-9
    args['y_symmetry'] = False
    args['x_symmetry'] = False

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args['young'],
        poisson=args['poisson'],
        device=device,
    ).double()

    cnn_kwargs = {
        'latent_size': latent_size,
        'dense_channels': dense_channels,
        'kernel_size': kernel_size,
        'conv_filters': conv_filters,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes = axes.flatten()

    first_stage_maxit = 1500

    # Trials and seeds
    seeds = [0, 10, 20, 30]
    for idx, seed in enumerate(seeds):
        ax = axes[idx]

        # Intialize random seed
        utils.build_random_seed(seed)
        cnn_kwargs['random_seed'] = seed

        model = models.MultiMaterialCNNModel(args, **cnn_kwargs).to(
            device=device, dtype=torch.double
        )

        # Calculate the initial compliance
        model.eval()
        with torch.no_grad():
            (
                initial_compliance,
                x_phys,
                _,
                _,
            ) = topo_physics.calculate_multi_material_compliance(
                model, ke, args, device, torch.double
            )

        # Detach calculation and use it for scaling in PyGranso
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0
        )

        # Train PyGranso MMTO - First Stage
        # Setup the combined function for PyGranso
        comb_fn = lambda model: train.multi_material_constraint_function(  # noqa
            model,
            initial_compliance,
            ke,
            args,
            device=device,
            dtype=torch.double,
        )

        train.train_pygranso_mmto(
            model=model, comb_fn=comb_fn, maxit=first_stage_maxit, device=device
        )

        # Get the final design
        (
            compliance,
            final_design,
            _,
            _,
        ) = topo_physics.calculate_multi_material_compliance(
            model, ke, args, device, torch.double
        )
        final_design = final_design.detach().numpy()

        design = final_design[:, 1].reshape(128, 128).T

        ax.imshow(design, cmap='Greys', aspect="auto")
        ax.set_title(f'Compliance = {compliance}')
        ax.set_xticks([])
        ax.set_yticks([])

    path = os.path.join(save_path, 'single-l-shape-results.png')
    fig.savefig(path)


@cli.command('test-single-material-mbb-beam')
def test_single_material_mbb_beam():
    # For testing, we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    penal = 3.0
    scale_factor = 1.0
    latent_size = 96
    dense_channels = 24
    kernel_size = (11, 11)
    ks_string = ",".join(map(str, kernel_size))

    # DIP Setup
    conv_filters = (256 // 2, 128 // 2, 64 // 2, 32 // 2)
    cv_string = ",".join(map(str, conv_filters))

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=[
            'topology-optimization-task',
            'single-material mbb-beam',
        ],
        config={
            'latent_size': latent_size,
            'dense_channels': dense_channels,
            'kernel_size': ks_string,
            'conv_filters': cv_string,
            'penal': penal,
            'scale_factor': scale_factor,
        },
    )

    # Create directory for saving the model and
    # output data
    save_path = os.path.join(
        '/users/5/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    device = torch.device('cpu')

    problem_name = 'mbb_beam_96x32_0.5'

    # Build the problems for pygranso and google
    PYGRANSO_PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)  # noqa

    # Build structure
    print(f"Building structure: {problem_name}")
    problem = PYGRANSO_PROBLEMS_BY_NAME.get(problem_name)

    args = topo_api.specified_task(problem, device=device)
    args['penal'] = penal
    args['scale_factor'] = scale_factor
    args['young_min'] = 1e-9
    args['y_symmetry'] = False
    args['x_symmetry'] = False

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args['young'],
        poisson=args['poisson'],
        device=device,
    ).double()

    cnn_kwargs = {
        'latent_size': latent_size,
        'dense_channels': dense_channels,
        'kernel_size': kernel_size,
        'conv_filters': conv_filters,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes = axes.flatten()

    first_stage_maxit = 1500

    # Trials and seeds
    seeds = [0, 10, 20, 30]
    for idx, seed in enumerate(seeds):
        ax = axes[idx]

        # Intialize random seed
        utils.build_random_seed(seed)
        cnn_kwargs['random_seed'] = seed

        model = models.MultiMaterialCNNModel(args, **cnn_kwargs).to(
            device=device, dtype=torch.double
        )

        # Calculate the initial compliance
        model.eval()
        with torch.no_grad():
            (
                initial_compliance,
                x_phys,
                _,
                _,
            ) = topo_physics.calculate_multi_material_compliance(
                model, ke, args, device, torch.double
            )

        # Detach calculation and use it for scaling in PyGranso
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0
        )

        # Train PyGranso MMTO - First Stage
        # Setup the combined function for PyGranso
        comb_fn = lambda model: train.multi_material_constraint_function(  # noqa
            model,
            initial_compliance,
            ke,
            args,
            device=device,
            dtype=torch.double,
        )

        train.train_pygranso_mmto(
            model=model, comb_fn=comb_fn, maxit=first_stage_maxit, device=device
        )

        # Get the final design
        (
            compliance,
            final_design,
            _,
            _,
        ) = topo_physics.calculate_multi_material_compliance(
            model, ke, args, device, torch.double
        )
        final_design = final_design.detach().numpy()

        design = final_design[:, 1].reshape(96, 32).T

        ax.imshow(design, cmap='Greys', aspect="auto")
        ax.set_title(f'Compliance = {compliance}')
        ax.set_xticks([])
        ax.set_yticks([])

    path = os.path.join(save_path, 'single-mbb-beam-results.png')
    fig.savefig(path)


if __name__ == "__main__":
    cli()
