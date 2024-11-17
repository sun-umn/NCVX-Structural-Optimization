# third party
import numpy as np
import scipy
import torch
import torch.nn as nn

# first party
import utils

# TODO: Investigate file


# Calculate the young modulus
def young_modulus(
    x, e_0, e_min, p=3, device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE
):
    """
    Function that calculates the young modulus
    """
    return (e_min + x**p * (e_0 - e_min)).to(device=device, dtype=dtype)


def young_modulus_multi_material_v2(
    x, e_materials, e_min, p=3, device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE
):
    """
    Function that calculates the young modulus for multiple materials
    """
    e_materials = e_materials[:, None, None]
    material_density = e_min + x**p * (e_materials - e_min)

    return material_density.sum(axis=0).to(device=device, dtype=dtype)


def young_modulus_multi_material(
    x, e_materials, e_min, p=3, device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE
):
    """
    Function that will compute the young's modulus for multiple materials
    """
    num_materials = len(e_materials)
    penalized_materials = x**p

    # Reshaping for matrix multiplication
    e_materials = e_materials.reshape(num_materials, 1).T
    material_density = (e_materials - e_min) * penalized_materials + e_min
    young_modulus = material_density.sum(axis=1).flatten()
    # import pdb; pdb.set_trace()

    # young_modulus = (e_materials * penalized_materials).sum(axis=1).flatten()
    return young_modulus.to(device=device, dtype=dtype)


# Define the physical density with torch
def physical_density(x, args, volume_constraint=True, filtering=False):
    """
    Function that calculates the physical density
    """
    shape = (args["nely"], args["nelx"])
    arg_mask = args["mask"]
    size_x = len(x.flatten())

    # In the code they do a reshape but this would not be necessary
    # if this assertion is broken
    assert x.shape == shape or x.ndim == 1

    if volume_constraint:
        if len(arg_mask.flatten()) == 1:
            x = utils.sigmoid_with_constrained_mean(x, args["volfrac"])

        else:
            mask = torch.broadcast_to(arg_mask, shape) > 0
            x = utils.sigmoid_with_constrained_mean(x[mask], args["volfrac"])
            flat_nonzero_mask = torch.nonzero(mask.ravel(), as_tuple=True)[0]
            x = utils.torch_scatter1d(x, flat_nonzero_mask, size_x)
            x = x.reshape(shape)

    else:
        x = x * args["mask"]

    if filtering:
        # TODO: there was a 2D gaussian filter here
        # but I removed it because I do not know
        # how it will react with pytorch
        pass

    return x


# Calculate the forces
def calculate_forces(x_phys, args):
    """
    Function that just gathers the current forces from the
    problem
    """
    # For now this is all that is happening in the code although
    # we should be aware of this if this could be an issue with
    # harder problems. This eliminates the gravitational force
    # on the object
    return args["forces"]


# Build a stiffness matrix
def get_stiffness_matrix(
    young: float, poisson: float, device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE
) -> torch.Tensor:
    """
    Function to build the elements of the stiffness matrix
    """
    # Build the element stiffness matrix
    # What are e & nu?
    _, nu = young, poisson

    A11 = torch.tensor(
        [[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]]
    ).to(device=device, dtype=dtype)
    A12 = torch.tensor(
        [[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]]
    ).to(device=device, dtype=dtype)
    B11 = torch.tensor(
        [[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]]
    ).to(device=device, dtype=dtype)
    B12 = torch.tensor(
        [[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]]
    ).to(device=device, dtype=dtype)

    M1 = torch.vstack([torch.hstack([A11, A12]), torch.hstack([A12.t(), A11])])
    M2 = torch.vstack([torch.hstack([B11, B12]), torch.hstack([B12.t(), B11])])

    ke = (1.0 / (1.0 - nu**2) / 24.0) * (M1 + nu * M2)

    return ke


# Compliance
def compliance(
    x_phys,
    u,
    ke,
    args,
    *,
    penal=3,
    e_min=1e-9,
    e_0=1,
    base="Google",
    device=utils.DEFAULT_DEVICE,
    dtype=utils.DEFAULT_DTYPE,
):
    """
    Calculate the compliance objective.
    NOTE: For our implementation both x_phys and u will require_grad
    and will both be torch tensors.
    """
    nely, nelx = args["nely"], args["nelx"]

    # Updated code
    edof, x_list, y_list = build_nodes_data(args, base=base)
    ce = (u[edof] @ ke) * u[edof]
    ce = torch.sum(ce, 1)
    ce = ce.reshape(nelx, nely)

    young_x_phys = young_modulus(
        x_phys, e_0, e_min, p=penal, device=device, dtype=dtype
    )

    return young_x_phys * ce.t(), None, None


def multi_material_compliance_v2(
    x_phys,
    u,
    ke,
    args,
    *,
    penal=3,
    e_min=1e-9,
    e_0=1,
    base="Google",
    device=utils.DEFAULT_DEVICE,
    dtype=utils.DEFAULT_DTYPE,
):
    """
    Calculate the compliance objective.
    NOTE: For our implementation both x_phys and u will require_grad
    and will both be torch tensors.
    """
    nely, nelx = args["nely"], args["nelx"]

    # Updated code
    edof, x_list, y_list = build_nodes_data(args, base=base)
    ce = (u[edof] @ ke) * u[edof]
    ce = torch.sum(ce, 1)
    ce = ce.reshape(nelx, nely)

    young_x_phys = young_modulus_multi_material_v2(
        x_phys, e_0, e_min, p=penal, device=device, dtype=dtype
    )

    return young_x_phys * ce.t(), None, None


def multi_material_compliance(
    stiffness,
    u,
    ke,
    args,
    *,
    penal=3,
    e_min=1e-9,
    e_0=1,
    base="Google",
    device=utils.DEFAULT_DEVICE,
    dtype=utils.DEFAULT_DTYPE,
):
    """
    Calculate the compliance objective.
    NOTE: For our implementation both x_phys and u will require_grad
    and will both be torch tensors.
    """
    # Updated code - for multi material compliance
    edof, x_list, y_list = build_nodes_data(args, base=base)
    ce = (u[edof] @ ke) * u[edof]
    ce = torch.sum(ce, axis=1)

    return stiffness * ce, None, None


def get_k_data(stiffness, ke, args, base="MATLAB"):
    """
    Function that is the pytorch version of get_K
    from the other repository
    """
    if not torch.is_tensor(ke):
        ke = torch.from_numpy(ke)

    edof, x_list, y_list = build_nodes_data(args, base=base)

    # stiffness flattened
    stiffness_flat = stiffness.t().flatten()
    stiffness_flat = stiffness_flat.reshape(1, len(stiffness_flat))

    # ke flattened
    ke_flat = ke.flatten()
    ke_flat = ke_flat.reshape(len(ke_flat), 1)

    # value list
    value_list = ke_flat @ stiffness_flat
    value_list = value_list.t().flatten()

    return value_list, y_list, x_list


def sparse_displace(
    x_phys,
    ke,
    args,
    forces,
    freedofs,
    fixdofs,
    *,
    penal=3,
    e_min=1e-9,
    e_0=1,
    base="Google",
    device=utils.DEFAULT_DEVICE,
    dtype=utils.DEFAULT_DTYPE,
):
    """
    Function that displaces the load x using finite element techniques.
    """
    stiffness = young_modulus(x_phys, e_0, e_min, p=penal, device=device, dtype=dtype)

    # Get the K values
    k_entries, k_ylist, k_xlist = get_k_data(stiffness, ke, args, base=base)
    k_ylist = k_ylist.to(device=device, dtype=dtype)
    k_xlist = k_xlist.to(device=device, dtype=dtype)

    index_map, keep, indices = utils._get_dof_indices(
        freedofs,
        fixdofs,
        k_ylist,
        k_xlist,
        k_entries,
    )

    # Reduced forces
    freedofs_forces = forces[freedofs].double().to(device=device, dtype=dtype)
    size = freedofs_forces.cpu().numpy().size

    # Require gradient on the forces
    freedofs_forces = freedofs_forces.requires_grad_()

    # Calculate u_nonzero
    keep_k_entries = k_entries[keep]

    # Build the sparse matrix
    K = torch.sparse_coo_tensor(indices, keep_k_entries, [size, size]).to(
        device=device, dtype=dtype
    )

    # Symmetric indices
    keep_k_entries = K.coalesce().values()
    indices = K.coalesce().indices()

    # Compute the u_matrix values
    u_nonzero = utils.solve_coo(
        keep_k_entries,
        indices,
        freedofs_forces,
        sym_pos=True,
        device=device,
        dtype=dtype,
    )
    fixdofs_zeros = torch.zeros(len(fixdofs)).to(device=device, dtype=dtype)
    u_values = torch.cat((u_nonzero, fixdofs_zeros))
    u_values = u_values[index_map].to(device=device, dtype=dtype)

    return u_values


def multi_material_sparse_displace_v2(
    x_phys,
    ke,
    args,
    forces,
    freedofs,
    fixdofs,
    *,
    penal=3,
    e_min=1e-9,
    e_0=1,
    base="Google",
    device=utils.DEFAULT_DEVICE,
    dtype=utils.DEFAULT_DTYPE,
):
    """
    Function that displaces the load x using finite element techniques.
    """
    stiffness = young_modulus_multi_material_v2(x_phys, e_0, e_min, p=penal, device=device, dtype=dtype)

    # Get the K values
    k_entries, k_ylist, k_xlist = get_k_data(stiffness, ke, args, base=base)
    k_ylist = k_ylist.to(device=device, dtype=dtype)
    k_xlist = k_xlist.to(device=device, dtype=dtype)

    index_map, keep, indices = utils._get_dof_indices(
        freedofs,
        fixdofs,
        k_ylist,
        k_xlist,
        k_entries,
    )

    # Reduced forces
    freedofs_forces = forces[freedofs].double().to(device=device, dtype=dtype)
    size = freedofs_forces.cpu().numpy().size

    # Require gradient on the forces
    freedofs_forces = freedofs_forces.requires_grad_()

    # Calculate u_nonzero
    keep_k_entries = k_entries[keep]

    # Build the sparse matrix
    K = torch.sparse_coo_tensor(indices, keep_k_entries, [size, size]).to(
        device=device, dtype=dtype
    )

    # Symmetric indices
    keep_k_entries = K.coalesce().values()
    indices = K.coalesce().indices()

    # Compute the u_matrix values
    u_nonzero = utils.solve_coo(
        keep_k_entries,
        indices,
        freedofs_forces,
        sym_pos=True,
        device=device,
        dtype=dtype,
    )
    fixdofs_zeros = torch.zeros(len(fixdofs)).to(device=device, dtype=dtype)
    u_values = torch.cat((u_nonzero, fixdofs_zeros))
    u_values = u_values[index_map].to(device=device, dtype=dtype)

    return u_values


def multi_material_sparse_displace(
    x_phys,
    ke,
    args,
    forces,
    freedofs,
    fixdofs,
    *,
    penal=3,
    e_min=1e-9,
    e_0=1,
    base="Google",
    device=utils.DEFAULT_DEVICE,
    dtype=utils.DEFAULT_DTYPE,
):
    """
    Function that displaces the load x using finite element techniques.
    """
    stiffness = x_phys

    # Get the K values
    k_entries, k_ylist, k_xlist = get_k_data(stiffness, ke, args, base=base)
    k_ylist = k_ylist.to(device=device, dtype=dtype)
    k_xlist = k_xlist.to(device=device, dtype=dtype)

    index_map, keep, indices = utils._get_dof_indices(
        freedofs,
        fixdofs,
        k_ylist,
        k_xlist,
        k_entries,
    )

    # Reduced forces
    freedofs_forces = forces[freedofs].double().to(device=device, dtype=dtype)
    size = freedofs_forces.cpu().numpy().size

    # Require gradient on the forces
    freedofs_forces = freedofs_forces.requires_grad_()

    # Calculate u_nonzero
    keep_k_entries = k_entries[keep]

    # Build the sparse matrix
    K = torch.sparse_coo_tensor(indices, keep_k_entries, [size, size]).to(
        device=device, dtype=dtype
    )

    # Symmetric indices
    keep_k_entries = K.coalesce().values()
    indices = K.coalesce().indices()

    # Compute the u_matrix values
    u_nonzero = utils.solve_coo(
        keep_k_entries,
        indices,
        freedofs_forces,
        sym_pos=True,
        device=device,
        dtype=dtype,
    )
    fixdofs_zeros = torch.zeros(len(fixdofs)).to(device=device, dtype=dtype)
    u_values = torch.cat((u_nonzero, fixdofs_zeros))
    u_values = u_values[index_map].to(device=device, dtype=dtype)

    return u_values


def calculate_compliance(model, ke, args, device, dtype):
    """
    Function to calculate the final compliance
    """
    logits = model(None)
    logits = logits.to(dtype=dtype)

    # kwargs for displacement
    kwargs = dict(
        penal=args["penal"],
        e_min=args["young_min"],
        e_0=args["young"],
        base="MATLAB",
        device=device,
        dtype=dtype,
    )
    x_phys = torch.sigmoid(logits)
    mask = torch.broadcast_to(args["mask"], x_phys.shape) > 0
    mask = mask.requires_grad_(False)
    x_phys = x_phys * mask.int()

    # Calculate the forces
    forces = calculate_forces(x_phys, args)

    # Calculate the u_matrix
    u_matrix = sparse_displace(
        x_phys, ke, args, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    # Calculate the compliance output
    compliance_output, _, _ = compliance(x_phys, u_matrix, ke, args, **kwargs)

    # The loss is the sum of the compliance
    return torch.sum(compliance_output), x_phys, mask


def calculate_multi_material_compliance_v2(model, ke, args, device, dtype):
    """
    Function to calculate the final compliance
    """
    logits = model(None)
    x_phys = logits.to(dtype=dtype)

    # kwargs for displacement
    kwargs = dict(
        penal=args["penal"],
        e_min=args["young_min"],
        e_0=args["young"],
        base="MATLAB",
        device=device,
        dtype=dtype,
    )

    # Calculate the forces
    forces = calculate_forces(x_phys, args)

    # Calculate the u_matrix
    u_matrix = multi_material_sparse_displace_v2(
        x_phys, ke, args, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    # Calculate the compliance output
    compliance_output, _, _ = multi_material_compliance_v2(
        x_phys, u_matrix, ke, args, **kwargs
    )

    # The loss is the sum of the compliance
    return torch.sum(compliance_output), x_phys, _


def calculate_multi_material_compliance(model, ke, args, device, dtype):
    """
    Function to calculate the final compliance
    """
    logits = model(None)
    x_phys = logits.to(dtype=dtype)

    # kwargs for displacement
    kwargs = dict(
        penal=args["penal"],
        e_min=args["young_min"],
        e_0=args["young"],
        base="MATLAB",
        device=device,
        dtype=dtype,
    )
    # print(kwargs)

    # For now set the mask to None
    mask = None

    # Need to compute a stiffness matrix
    stiffness = young_modulus_multi_material(
        x_phys[:, 1:],
        e_materials=args['e_materials'],
        e_min=args['young_min'],
        p=args['penal'],
        device=device,
        dtype=dtype,
    )

    # Calculate the forces
    forces = calculate_forces(x_phys, args)

    # Calculate the u_matrix
    u_matrix = multi_material_sparse_displace(
        stiffness, ke, args, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    # Calculate the compliance output
    compliance_output, _, _ = multi_material_compliance(
        stiffness, u_matrix, ke, args, **kwargs
    )

    # The loss is the sum of the compliance
    return torch.sum(compliance_output), x_phys, mask


def build_K_matrix(x_phys, args, base="MATLAB"):
    """
    Function to build the full K matrix
    """
    # Build the stiffness using young's modulus
    stiffness = young_modulus(
        x_phys, e_0=args["young"], e_min=args["young_min"], p=args["penal"]
    )

    # Build KE
    ke = get_stiffness_matrix(young=args["young"], poisson=args["poisson"])

    # Get the K indices and values
    k_entries, k_ylist, k_xlist = get_k_data(
        stiffness=stiffness, ke=ke, args=args, base=base
    )

    # Create full indices
    if base == "MATLAB":
        indices = torch.stack([k_ylist, k_xlist])
        k_entries = k_entries

        # Create the K matrix
        K = (torch.sparse_coo_tensor(indices, k_entries)).to_dense().double()
        K = (K + K.transpose(1, 0)) / 2.0

    elif base == "Google":
        freedofs = args["freedofs"].numpy()
        fixdofs = args["fixdofs"].numpy()
        free_forces = args["forces"][freedofs].numpy()

        # Get the indices
        index_map, keep, indices = utils._get_dof_indices(
            freedofs, fixdofs, k_ylist, k_xlist, k_entries
        )
        size = free_forces.size

        # Calculate the K matrix for the google based code
        indices = indices.numpy()

        # Compute K
        K = (
            scipy.sparse.coo_matrix((k_entries[keep], indices), shape=(size,) * 2)
            .tocsc()
            .todense()
        )
        K = (K + K.T) / 2.0

    else:
        raise ValueError("Only methods MATLAB and Google are valid!")

    return K


def build_nodes_data(args, base="MATLAB"):
    """
    Build the nodes matrix from both the MATLAB and Google
    code. Turns out the code from the two different methods is
    different but we want to be able to test that we have
    the right implementation for each method. These same nodes will
    also be fed into the compliance objective.
    """
    nely, nelx = args["nely"], args["nelx"]

    # Compute the torch based meshgrid
    ely, elx = torch.meshgrid(torch.arange(nely), torch.arange(nelx))
    ely, elx = ely.transpose(1, 0), elx.transpose(1, 0)
    ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)

    # Calculate nodes - reflects the MATLAB code
    if base == "MATLAB":
        n1 = (nely + 1) * (elx + 0) + (ely + 1)
        n2 = (nely + 1) * (elx + 1) + (ely + 1)
        n3 = (nely + 1) * (elx + 1) + (ely + 0)
        n4 = (nely + 1) * (elx + 0) + (ely + 0)

    elif base == "Google":
        # Google implementation
        n1 = (nely + 1) * (elx + 0) + (ely + 0)
        n2 = (nely + 1) * (elx + 1) + (ely + 0)
        n3 = (nely + 1) * (elx + 1) + (ely + 1)
        n4 = (nely + 1) * (elx + 0) + (ely + 1)

    else:
        raise ValueError("Only options are MATLAB and Google!")

    # The shape of this matrix results in
    # (8, nelx, nely)
    edof = torch.stack(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
    )

    # Calculate the transpose & get the first
    # element
    edof = edof.permute(2, 1, 0)[0]

    # Build the i & j indices for a sparse matrix
    y_list = edof.tile((8,)).flatten()
    x_list = edof.repeat_interleave(8)

    return edof, y_list, x_list


def build_nodes_data_google(args):
    """
    An exact implementation of the Google code to build
    the nodes for K and the compliance
    """
    nely, nelx = args["nely"], args["nelx"]

    # get position of the nodes of each element in the stiffness matrix
    ely, elx = np.meshgrid(range(nely), range(nelx))  # x, y coords
    ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)

    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)
    edof = np.array(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
    )
    edof = edof.T[0]

    return edof


def get_stiffness_matrix_google(young, poisson):
    """
    Function from the google code to compute the stiffness
    matrix
    """
    # Element stiffness matrix
    e, nu = young, poisson
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    return (
        e
        / (1 - nu**2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )
