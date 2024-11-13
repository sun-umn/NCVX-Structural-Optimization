from typing import Any, Dict

# third party
import numpy as np
import torch

from utils import DEFAULT_DEVICE, DEFAULT_DTYPE

X, Y = 0, 1


def specified_task(problem, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
    """
    Given a problem, return parameters for
    running a topology optimization.

    NOTE: Based on what I have been learning about pytorch
    we may need to update these inputs to be torch tensors.

    NOTE: Nothing to check here
    """
    fixdofs = np.flatnonzero(problem.normals.ravel().cpu().detach().clone())
    alldofs = np.arange(2 * (problem.width + 1) * (problem.height + 1))
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

    # Variables that will utilize GPU calculations
    mask = torch.tensor(problem.mask).to(device=device, dtype=dtype)
    freedofs = torch.tensor(freedofs).to(device=device, dtype=torch.long)
    fixdofs = torch.tensor(fixdofs).to(device=device, dtype=torch.long)

    params = {
        # material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # constraints
        "volfrac": problem.density,
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": torch.tensor(problem.width),
        "nely": torch.tensor(problem.height),
        "mask": mask,
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": problem.forces.ravel(),
        "penal": 3.0,
        "filter_width": 2,
        "epsilon": problem.epsilon,
        'ndof': len(alldofs),
        'tounn_mask': problem.tounn_mask,
    }
    return params


def multi_material_mbb_beam(
    nelx: int,
    nely: int,
    e_materials: torch.Tensor,
    material_density_weight: torch.Tensor,
    combined_frac: float,
    epsilon=1e-5,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
) -> Dict[str, Any]:
    """Textbook beam example."""
    ndof = 2 * (nelx + 1) * (nely + 1)

    normals = torch.zeros((nely + 1, nelx + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, -1, Y] = 1
    normals[0, :, X] = 1

    forces = torch.zeros((nely + 1, nelx + 1, 2)).to(device=device, dtype=dtype)
    forces[0, 0, Y] = -1

    fixdofs = np.flatnonzero(normals.ravel().cpu().detach().clone())
    alldofs = np.arange(2 * (nely + 1) * (nelx + 1))
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

    # Variables that will utilize GPU calculations
    mask = torch.tensor(1.0).to(device=device, dtype=dtype)
    freedofs = torch.tensor(freedofs).to(device=device, dtype=torch.long)
    fixdofs = torch.tensor(fixdofs).to(device=device, dtype=torch.long)

    params = {
        # material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # constraints
        "combined_frac": combined_frac,
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": torch.tensor(nelx),
        "nely": torch.tensor(nely),
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": forces,
        "penal": 3.0,
        "filter_width": 2,
        "epsilon": epsilon,
        "ndof": ndof,
        "e_materials": e_materials,
        "material_density_weight": material_density_weight,
    }
    return params


def multi_material_suspended_bridge(
    nelx,
    nely,
    e_materials: torch.Tensor,
    material_density_weight: torch.Tensor,
    combined_frac: float,
    epsilon=1e-5,
    span_position=0.2,
    anchored=False,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A bridge above the ground, with supports at lower corners."""
    ndof = 2 * (nelx + 1) * (nely + 1)

    normals = torch.zeros((nelx + 1, nely + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, :, X] = 1
    normals[: round(span_position * nelx), -1, Y] = 1
    if anchored:
        normals[: round(span_position * nelx), -1, X] = 1

    forces = torch.zeros((nelx + 1, nely + 1, 2)).to(device=device, dtype=dtype)
    forces[:, -1, Y] = -1 / nelx

    fixdofs = np.flatnonzero(normals.ravel().cpu().detach().clone())
    alldofs = np.arange(2 * (nely + 1) * (nelx + 1))
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

    # Variables that will utilize GPU calculations
    mask = torch.tensor(1.0).to(device=device, dtype=dtype)
    freedofs = torch.tensor(freedofs).to(device=device, dtype=torch.long)
    fixdofs = torch.tensor(fixdofs).to(device=device, dtype=torch.long)

    params = {
        # material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # constraints
        "combined_frac": combined_frac,
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": torch.tensor(nelx),
        "nely": torch.tensor(nely),
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": forces,
        "penal": 3.0,
        "filter_width": 2,
        "epsilon": epsilon,
        "ndof": ndof,
        "e_materials": e_materials,
        "material_density_weight": material_density_weight,
    }
    return params


def multi_material_tip_cantilever_task(
    nelx: int,
    nely: int,
    e_materials: torch.Tensor,
    material_density_weight: torch.Tensor,
    combined_frac: float,
    epsilon=1e-3,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
) -> Dict[str, Any]:
    """
    Function that will create the design space for the multi
    material tip cantilever structure
    """
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Forces on the system
    forces = torch.zeros((ndof, 1))
    forces[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0] = -1

    # Degrees of freedom
    alldofs_array = np.arange(ndof)

    # Fixed dofs
    fixdofs_array = alldofs_array[0 : 2 * (nely + 1) : 1]

    # Free dofs
    freedofs_array = np.sort(list(set(alldofs_array) - set(fixdofs_array)))

    # Convert to torch tensorse)
    freedofs = torch.tensor(freedofs_array).to(device=device, dtype=torch.long)
    fixdofs = torch.tensor(fixdofs_array).to(device=device, dtype=torch.long)

    params = {
        # material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # constraints
        "combined_frac": combined_frac,
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": torch.tensor(nelx),
        "nely": torch.tensor(nely),
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": forces,
        "penal": 3.0,
        "filter_width": 2,
        "epsilon": epsilon,
        "ndof": len(alldofs_array),
        "e_materials": e_materials,
        "material_density_weight": material_density_weight,
    }
    return params


def multi_material_bridge_task(
    nelx: int,
    nely: int,
    e_materials: torch.Tensor,
    material_density_weight: torch.Tensor,
    combined_frac: float,
    epsilon=1e-3,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
) -> Dict[str, Any]:
    """
    Function that will create the design space for the multi
    material tip cantilever structure
    """
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Forces on the system
    forces = torch.zeros((ndof, 1))

    forces[2 * (nely + 1) * int(nelx // 4 + 1) - 1, 0] = -1
    forces[2 * (nely + 1) * int(2 * nelx // 4 + 1) - 1, 0] = -2
    forces[2 * (nely + 1) * int(3 * nelx // 4 + 1) - 1, 0] = -1

    # Degrees of freedom
    alldofs_array = np.arange(ndof)

    # Fixed dofs
    fixdofs_array = np.union1d(
        np.array([2 * (nely + 1) - 1 - 1, 2 * (nely + 1) - 1]),
        np.array([2 * (nelx + 1) * (nely + 1) - 1]),
    )

    # Free dofs
    freedofs_array = np.sort(list(set(alldofs_array) - set(fixdofs_array)))

    # Convert to torch tensorse)
    freedofs = torch.tensor(freedofs_array).to(device=device, dtype=torch.long)
    fixdofs = torch.tensor(fixdofs_array).to(device=device, dtype=torch.long)

    params = {
        # material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # constraints
        "combined_frac": combined_frac,
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": torch.tensor(nelx),
        "nely": torch.tensor(nely),
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": forces,
        "penal": 3.0,
        "filter_width": 2,
        "epsilon": epsilon,
        "ndof": len(alldofs_array),
        "e_materials": e_materials,
        "material_density_weight": material_density_weight,
    }
    return params
