import numpy as np

def yaw_from_quaternion(wxyz):
    # wxyz: [w,x,y,z] -> yaw (around z)
    w,x,y,z = wxyz
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def transform_points(points_xyz, R, t):
    # points: [N,3], R: [3,3], t: [3]
    return (points_xyz @ R.T) + t

def to_hom(points_xy):
    # [N,2] -> [N,3]
    return np.concatenate([points_xy, np.ones((points_xy.shape[0],1))], axis=1)
