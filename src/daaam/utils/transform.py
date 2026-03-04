import numpy as np


def quaternion_to_angular_velocity(q1, q2, dt):
    """
    Compute angular velocity from two quaternions.
    
    Parameters:
    -----------
    q1 : array-like, shape (4,)
        Initial quaternion [qx, qy, qz, qw]
    q2 : array-like, shape (4,)
        Final quaternion [qx, qy, qz, qw]
    dt : float
        Time difference between q1 and q2
    
    Returns:
    --------
    omega : np.ndarray, shape (3,)
        Angular velocity vector [wx, wy, wz] in the same frame as quaternions
    """
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)
    
    # Normalize quaternions to ensure unit quaternions
    q1 = q1 / (np.linalg.norm(q1) + 1e-10)
    q2 = q2 / (np.linalg.norm(q2) + 1e-10)
    
    # Method 1: Using quaternion difference (most common and stable)
    return angular_velocity_from_quaternion_difference(q1, q2, dt)


def angular_velocity_from_quaternion_difference(q1, q2, dt):
    """
    Compute angular velocity using quaternion difference method.
    This is the most numerically stable for small time steps.
    """
    # Compute relative quaternion: q_diff = q2 * q1^(-1)
    q1_inv = quaternion_conjugate(q1)
    q_diff = quaternion_multiply(q2, q1_inv)
    
    # Ensure shortest path rotation (handle quaternion double cover)
    if q_diff[3] < 0:
        q_diff = -q_diff
    
    # Extract angle-axis from quaternion difference
    angle = 2 * np.arccos(np.clip(q_diff[3], -1.0, 1.0))
    
    if angle < 1e-6:  # Small angle approximation
        # For small angles, use linear approximation to avoid division by zero
        omega = 2 * q_diff[:3] / dt
    else:
        # Extract axis from quaternion
        sin_half_angle = np.sin(angle / 2)
        if abs(sin_half_angle) > 1e-6:
            axis = q_diff[:3] / sin_half_angle
        else:
            axis = q_diff[:3] / (np.linalg.norm(q_diff[:3]) + 1e-10)
        
        # Angular velocity
        omega = axis * angle / dt
    
    return omega

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q1 * q2
    Quaternions as [qx, qy, qz, qw]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_conjugate(q):
    """
    Compute quaternion conjugate (inverse for unit quaternions).
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.
    """
    x, y, z, w = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R

def compute_ema_velocity(transforms, timestamps, alpha=0.6):
    """
    Exponential moving average velocity estimation for online/real-time applications
    
    alpha: Smoothing factor (0 < alpha < 1)
        Higher = more weight on recent samples

    returns:
        twist: np.ndarray or None
            twist vector with time stamp [t, vx, vy, vz, wx, wy, wz]
            Returns None if velocity cannot be computed (only one transform)
    """
    prev_transform = None
    prev_time = None
    smoothed_v = None
    smoothed_omega = None
    velocity = None
    
    for transform, t in zip(transforms, timestamps):
        if prev_transform is not None:
            dt = t - prev_time
            
            # avoid division by zero
            if dt <= 0:
                continue
            
            v_instant = (transform[:3] - prev_transform[:3]) / dt
            omega_instant = quaternion_to_angular_velocity(
                prev_transform[3:7], transform[3:7], dt
            )
            
            # EMA
            if smoothed_v is None:
                smoothed_v = v_instant
                smoothed_omega = omega_instant
            else:
                smoothed_v = alpha * v_instant + (1 - alpha) * smoothed_v
                smoothed_omega = alpha * omega_instant + (1 - alpha) * smoothed_omega
            
            velocity = np.concatenate([[t], smoothed_v, smoothed_omega])
        
        prev_transform = transform
        prev_time = t
    
    return velocity