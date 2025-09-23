import numpy as np

def yaw_from_quaternion(q):
    """q = [w, x, y, z] -> yaw (rad, z-axis)"""
    w, x, y, z = q
    return np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

def se3(R, t):
    """R[3,3], t[3] -> T[4,4]"""
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def invert_se3(T):
    R = T[:3,:3]; t = T[:3,3]
    Ri = R.T
    ti = -Ri @ t
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3,:3] = Ri
    Ti[:3,3] = ti
    return Ti

def compose_se3(A, B):
    """A@B (both 4x4)"""
    return A @ B

def transform_points(X, R, t):
    """X[N,3] * R^T + t -> Y[N,3]"""
    return (X @ R.T) + t

def to_hom(X):
    """[N,2 or 3] -> [N,3 or 4]"""
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.concatenate([X, ones], axis=1)

def from_hom(Xh):
    """[N,3 or 4] -> [N,2 or 3]"""
    return Xh[:, :-1] / Xh[:, -1:]    
