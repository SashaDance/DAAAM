import numpy as np
import struct
import collections

# Based on COLMAP's read_write_model.py https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/scripts/python/read_write_model.py
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])

CAMERA_MODELS = {
    0: CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    1: CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    2: CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    3: CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    4: CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    5: CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    6: CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    7: CameraModel(model_id=7, model_name="FOV", num_params=5),
    8: CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    9: CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    10: CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}


def read_next_bytes(fid, num_bytes, format_char_sequence):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """Read COLMAP camera parameters from a binary file."""
    cameras = {}

    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODELS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)

            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_id,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
        return cameras


def camera_to_camera_info(camera):
    """Convert a COLMAP camera to a ROS CameraInfo format."""
    model = camera.model
    params = camera.params
    width = camera.width
    height = camera.height

    K = np.zeros(9)  # 3x3 camera matrix
    D = np.zeros(4)  # distortion parameters
    R = np.eye(3).flatten()  # rotation matrix (identity)
    P = np.zeros(12)  # projection matrix

    if model == 0:  # SIMPLE_PINHOLE
        # params: f, cx, cy
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]

    elif model == 1:  # PINHOLE
        # params: fx, fy, cx, cy
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]

    elif model in [2, 3, 8, 9]:  # radial distortion
        # For SIMPLE_RADIAL, RADIAL, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]

    elif model == 4:  # OPENCV
        # params: fx, fy, cx, cy, k1, k2, p1, p2
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        D[0] = params[4]  # k1
        D[1] = params[5]  # k2
        D[2] = params[6]  # p1
        D[3] = params[7]  # p2

    # Set intrinsic matrix K
    K[0] = fx  # fx
    K[4] = fy  # fy
    K[2] = cx  # cx
    K[5] = cy  # cy
    K[8] = 1.0  # 1

    # Set projection matrix P (first 3x3 part is K)
    P[0] = fx
    P[5] = fy
    P[2] = cx
    P[6] = cy
    P[10] = 1.0

    return {
        "height": height,
        "width": width,
        "distortion_model": "radial-tangential",
        "D": D.tolist(),
        "K": K.tolist(),
        "R": R.tolist(),
        "P": P.tolist(),
    }
