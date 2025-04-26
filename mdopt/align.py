import numpy as np
from typing import Any, Optional, Tuple
from numpy import ndarray


def kabsch_weighted(
    P: ndarray, Q: ndarray, W: Optional[ndarray] = None
) -> Tuple[ndarray, ndarray, float]:
    """
    Using the Kabsch algorithm with two sets of paired point P and Q.
    Each vector set is represented as an NxD matrix, where D is the
    dimension of the space.
    An optional vector of weights W may be provided.

    Note that this algorithm does not require that P and Q have already
    been overlayed by a centroid translation.

    The function returns the rotation matrix U, translation vector V,
    and RMS deviation between Q and P', where P' is:

        P' = P * U + V

    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.

    Returns
    -------
    U    : matrix
           Rotation matrix (D,D)
    V    : vector
           Translation vector (D)
    RMSD : float
           Root mean squared deviation between P and Q
    """
    # Computation of the weighted covariance matrix
    CMP = np.zeros(3)
    CMQ = np.zeros(3)
    C = np.zeros((3, 3))
    if W is None:
        W = np.ones(len(P)) / len(P)
    W = np.array([W, W, W]).T
    # NOTE UNUSED psq = 0.0
    # NOTE UNUSED qsq = 0.0
    iw = 3.0 / W.sum()
    n = len(P)
    for i in range(3):
        for j in range(n):
            for k in range(3):
                C[i, k] += P[j, i] * Q[j, k] * W[j, i]
    CMP = (P * W).sum(axis=0)
    CMQ = (Q * W).sum(axis=0)
    PSQ = (P * P * W).sum() - (CMP * CMP).sum() * iw
    QSQ = (Q * Q * W).sum() - (CMQ * CMQ).sum() * iw
    C = (C - np.outer(CMP, CMQ) * iw) * iw

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U, translation vector V, and calculate RMSD:
    U = np.dot(V, W)
    msd = (PSQ + QSQ) * iw - 2.0 * S.sum()
    if msd < 0.0:
        msd = 0.0
    rmsd_ = np.sqrt(msd)
    V = np.zeros(3)
    for i in range(3):
        t = (U[i, :] * CMQ).sum()
        V[i] = CMP[i] - t
    V = V * iw
    return U, V, rmsd_


def centroid(X: ndarray) -> ndarray:
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    https://en.wikipedia.org/wiki/Centroid

    C = sum(X)/len(X)

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    C : ndarray
        centroid
    """
    C: ndarray = X.mean(axis=0)
    return C



def kabsch_weighted_fit(
    P: ndarray,
    Q: ndarray,
    W: Optional[ndarray] = None,
    return_rmsd: bool = False,
) -> Tuple[ndarray, Optional[float]]:
    """
    Fit P to Q with optional weights W.
    Also returns the RMSD of the fit if return_rmsd=True.

    Parameters
    ----------
    P    : array
           (N,D) matrix, where N is points and D is dimension.
    Q    : array
           (N,D) matrix, where N is points and D is dimension.
    W    : vector
           (N) vector, where N is points
    rmsd : Bool
           If True, rmsd is returned as well as the fitted coordinates.

    Returns
    -------
    P'   : array
           (N,D) matrix, where N is points and D is dimension.
    RMSD : float
           if the function is called with rmsd=True
    """
    rmsd_: float
    R, T, rmsd_ = kabsch_weighted(Q, P, W)
    PNEW: ndarray = np.dot(P, R.T) + T
    if return_rmsd:
        return (PNEW, rmsd_)

    return (PNEW, None)