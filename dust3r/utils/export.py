# Adapted from https://github.com/nerlfield/wild-gaussian-splatting/blob/9d51831673666c08c79caef02286bebd2566f019/notebooks/00_dust3r_inference.ipynb
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.utils.device import to_numpy


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat

    K = np.asarray(
        [
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
        ]
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def init_filestructure(save_path):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    images_path = save_path / "images"
    masks_path = save_path / "masks"
    sparse_path = save_path / "sparse" / "0"

    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)
    sparse_path.mkdir(exist_ok=True, parents=True)

    return save_path, images_path, masks_path, sparse_path


def save_images_masks(imgs, masks, images_path: Path, masks_path: Path):
    # Saving images and optionally masks/depth maps
    for i, (image, mask) in enumerate(zip(imgs, masks)):
        image_save_path = images_path / f"{i}.png"

        mask_save_path = masks_path / f"{i}.png"
        image[~mask] = 1.
        rgb_image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)

        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2) * 255
        Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)


def save_cameras(focals, principal_points, sparse_path: Path, imgs_shape):
    # Save cameras.txt
    cameras_file = sparse_path / "cameras.txt"
    with cameras_file.open("w") as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            cameras_file.write(f"{i} PINHOLE {imgs_shape[2]} {imgs_shape[1]} {focal[0]} {focal[0]} {pp[0]} {pp[1]}\n")


def save_imagestxt(world2cam, sparse_path: Path):
    # Save images.txt
    images_file = sparse_path / "images.txt"
    # Generate images.txt content
    with images_file.open("w") as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(world2cam.shape[0]):
            # Convert rotation matrix to quaternion
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            images_file.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {i}.png\n")
            images_file.write("\n")  # Placeholder for points, assuming no points are associated with images here


def save_pointcloud_with_normals(imgs, pts3d, msk, sparse_path: Path):
    pc = get_pc(imgs, pts3d, msk)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    save_path = sparse_path / "points3D.ply"

    # Construct the header of the PLY file
    header = (
        f"ply\n"
        f"format ascii 1.0\n"
        f"element vertex {len(vertices)}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"property float nx\n"
        f"property float ny\n"
        f"property float nz\n"
        f"end_header\n"
    )

    # Write the PLY file
    with save_path.open('w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write(
                f"{vertex[0]} {vertex[1]} {vertex[2]} {int(color[0])} {int(color[1])} {int(color[2])} {normal[0]} {normal[1]} {normal[2]}\n")


import trimesh


def get_pc(imgs, pts3d, mask):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)

    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])

    pts = pts.reshape(-1, 3)[::3]
    col = col.reshape(-1, 3)[::3]

    # mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))

    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals  # Manually add normals to the point cloud

    return pct  # , pts


def save_to_colmap(save_dir: Path, scene: BasePCOptimizer, *, min_conf_thr: float = 20):
    # Extract information from scene
    world2cam = np.linalg.inv(scene.get_im_poses().detach().cpu().numpy())
    principal_points = scene.get_principal_points().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    if len(focals.shape) != 2:
        focals = focals[:, None]
    imgs = np.array(scene.imgs)
    pts3d = [i.detach() for i in scene.get_pts3d()]

    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    masks = to_numpy(scene.get_masks())

    # Save stuff
    save_path, images_path, masks_path, sparse_path = init_filestructure(save_dir)
    save_images_masks(imgs, masks, images_path, masks_path)
    save_cameras(focals, principal_points, sparse_path, imgs_shape=imgs.shape)
    save_imagestxt(world2cam, sparse_path)
    save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)
