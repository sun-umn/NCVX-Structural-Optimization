import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def ordered_simp_interpolation(nelx, nely, x, penal, X, Y):
    y = np.zeros((nely, nelx))
    dy = np.zeros((nely, nelx))

    for i in range(nelx):
        for j in range(nely):
            for k in range(len(X) - 1):
                if (X[k] < x[j, i]) and (X[k + 1] >= x[j, i]):
                    A = (Y[k] - Y[k + 1]) / (
                        X[k] ** (1 * penal) - X[k + 1] ** (1 * penal)
                    )
                    B = Y[k] - A * (X[k] ** (1 * penal))
                    y[j, i] = A * (x[j, i] ** (1 * penal)) + B
                    dy[j, i] = A * penal * (x[j, i] ** ((1 * penal) - 1))
                    break

    return y, dy


# TODO: Will want to extract the force calculations from this part of the function
def finite_element(args, nelx, nely, E_Interpolation, KE):
    K = lil_matrix((2 * (nelx + 1) * (nely + 1), 2 * (nelx + 1) * (nely + 1)))
    F = lil_matrix((2 * (nely + 1) * (nelx + 1), 1))
    U = np.zeros((2 * (nely + 1) * (nelx + 1), 1))

    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = (
                np.array(
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
                + 1
            )
            K[edof[:, np.newaxis], edof] += E_Interpolation[ely, elx] * KE

    # Define loads (F) and supports (fixeddofs)
    F = lil_matrix(args['forces']).T
    fixeddofs = args['fixdofs']
    freedofs = args['freedofs']

    # Solving for displacements
    U[freedofs, 0] = spsolve(
        K.tocsr()[freedofs, :].tocsc()[:, freedofs], F[freedofs, 0].toarray()
    )

    # Applying boundary conditions
    U[fixeddofs, 0] = 0

    return U


def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros((nely, nelx))

    for i in range(nelx):
        for j in range(nely):
            total_sum = 0.0
            for k in range(
                max(i - int(np.floor(rmin)), 0), min(i + int(np.floor(rmin)) + 1, nelx)
            ):
                for l in range(  # noqa
                    max(j - int(np.floor(rmin)), 0),
                    min(j + int(np.floor(rmin)) + 1, nely),
                ):
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    total_sum += max(0, fac)
                    dcn[j, i] += max(0, fac) * x[l, k] * dc[l, k]

            if total_sum > 0:
                dcn[j, i] /= x[j, i] * total_sum

    return dcn


def optimality_criterion(
    nelx, nely, x, volfrac, costfrac, dc, E_, dE_, P_, dP_, loop, MinMove
):
    dc = -1 * dc
    lV1 = 0
    lV2 = 2 * np.max(dc)
    Temp = P_ + x * dP_
    Temp = dc / Temp
    lP1 = 0
    lP2 = 2 * np.max(Temp)
    move = max(0.15 * 0.96**loop, MinMove)

    while ((lV2 - lV1) / (lV1 + lV2) > 1e-6) or ((lP2 - lP1) / (lP1 + lP2) > 1e-6):
        lmidV = 0.5 * (lV2 + lV1)
        lmidP = 0.5 * (lP2 + lP1)
        Temp = lmidV + lmidP * P_ + lmidP * x * dP_
        Coef = dc / Temp
        Coef = np.abs(Coef)
        xnew = np.maximum(
            10**-5,
            np.maximum(
                x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(Coef)))
            ),
        )

        if np.sum(xnew) - volfrac * nelx * nely > 0:
            lV1 = lmidV
        else:
            lV2 = lmidV

        CurrentCostFrac = np.sum(xnew * P_) / (nelx * nely)

        if CurrentCostFrac - costfrac > 0:
            lP1 = lmidP
        else:
            lP2 = lmidP

    return xnew
