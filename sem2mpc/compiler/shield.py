import casadi as ca

def soft_barrier(distance_sq, radius_sq, w=5.0, eps=1e-4):
    """Soft CBF-like barrier: -w * log(distance^2 - r^2 + eps)"""
    return -w * ca.log(distance_sq - radius_sq + eps)

def min_distance_sq_along_traj(X, center):
    """Compute min squared distance of positions in X to obstacle center (numpy later)."""
    dx = X[0, :] - center[0]
    dy = X[1, :] - center[1]
    return ca.mmin(dx*dx + dy*dy)
