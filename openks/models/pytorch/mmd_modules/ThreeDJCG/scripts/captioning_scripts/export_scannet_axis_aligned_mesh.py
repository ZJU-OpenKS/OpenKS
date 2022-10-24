import os
import sys

import numpy as np

from tqdm import tqdm
from plyfile import PlyData, PlyElement

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from lib.config import CONF

SCANNET_LIST = [s.rstrip() for s in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")).readlines() if s.rstrip().split("_")[-1] == "00"]
SCANNET_LIST = [s for s in SCANNET_LIST if int(s.split("_")[0][5:]) < 707]
SCANNET_MESH = os.path.join(CONF.PATH.SCANNET_SCANS, "{}/{}_vh_clean_2.ply") # scene_id, scene_id
SCANNET_META = os.path.join(CONF.PATH.SCANNET_SCANS, "{}/{}.txt") # scene_id, scene_id

OUTPUT_PATH = CONF.PATH.AXIS_ALIGNED_MESH

os.makedirs(OUTPUT_PATH, exist_ok=True)

def read_mesh(filename):
    """ read XYZ for each vertex.
    """

    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']

    return vertices, plydata['face']

def export_mesh(vertices, faces):
    new_vertices = []
    for i in range(vertices.shape[0]):
        new_vertices.append(
            (
                vertices[i][0],
                vertices[i][1],
                vertices[i][2],
                vertices[i][3],
                vertices[i][4],
                vertices[i][5],
            )
        )

    vertices = np.array(
        new_vertices,
        dtype=[
            ("x", np.dtype("float32")), 
            ("y", np.dtype("float32")), 
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8"))
        ]
    )

    vertices = PlyElement.describe(vertices, "vertex")
    
    return PlyData([vertices, faces])

def align_mesh(scene_id, mesh_path):
    vertices, faces = read_mesh(mesh_path)
    for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
        if 'axisAlignment' in line:
            axis_align_matrix = np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
            break
    
    # align
    pts = np.ones((vertices.shape[0], 4))
    pts[:, :3] = vertices[:, :3]
    pts = np.dot(pts, axis_align_matrix.T)
    vertices[:, :3] = pts[:, :3]

    mesh = export_mesh(vertices, faces)

    return mesh

print("aligning ScanNet meshes...")
for scene_id in tqdm(SCANNET_LIST):
    scene_path = os.path.join(OUTPUT_PATH, scene_id)
    os.makedirs(scene_path, exist_ok=True)

    mesh = align_mesh(scene_id, os.path.join(SCANNET_MESH.format(scene_id, scene_id)))
    mesh.write(os.path.join(scene_path, "axis_aligned_scene.ply"))

print("done!")
