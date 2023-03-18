import os
import numpy as np

def load_poses(pose_path):
  poses = []
  try:
    if '.txt' in pose_path:
      with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
          T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
          T_w_cam0 = T_w_cam0.reshape(3, 4)
          T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
          poses.append(T_w_cam0)
    else:
      poses = np.load(pose_path)['arr_0']
  
  except FileNotFoundError:
    print('Ground truth poses are not avaialble.')
  
  return np.array(poses)


def load_calib(calib_path):
  T_cam_velo = []
  try:
    with open(calib_path, 'r') as f:
      lines = f.readlines()
      for line in lines:
        if 'Tr:' in line:
          line = line.replace('Tr:', '')
          T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
          T_cam_velo = T_cam_velo.reshape(3, 4)
          T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
  
  except FileNotFoundError:
    print('Calibrations are not avaialble.')
  
  return np.array(T_cam_velo)


def range_projection(current_vertex, fov_up=10.67, fov_down=-30.67, proj_H=32, proj_W=900, max_range=80, cut_range=True,
                     lower_bound=0.1, upper_bound=6):

  fov_up = fov_up / 180.0 * np.pi
  fov_down = fov_down / 180.0 * np.pi
  fov = abs(fov_down) + abs(fov_up)

  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)

  if cut_range:
    current_vertex = current_vertex[
      (depth > lower_bound) & (depth < upper_bound)]
    depth = depth[(depth > lower_bound) & (depth < upper_bound)]
  else:
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]
    depth = depth[(depth > 0) & (depth < max_range)]

  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]

  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)

  proj_x = 0.5 * (yaw / np.pi + 1.0)
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov

  proj_x *= proj_W
  proj_y *= proj_H

  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)

  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)

  order = np.argsort(depth)[::-1]
  depth = depth[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]

  indices = np.arange(depth.shape[0])
  indices = indices[order]

  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)
  proj_idx = np.full((proj_H, proj_W), -1,
                     dtype=np.int32)

  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
  proj_idx[proj_y, proj_x] = indices

  return proj_range, proj_vertex, proj_idx

def bev_projection(current_vertex, proj_H=32, proj_W=900, max_range=80, cut_height=True, lower_bound=10, upper_bound=20):

  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
  scan_z_tmp = current_vertex[:, 2]
  if cut_height:
    current_vertex = current_vertex[(depth > 0) & (depth < max_range) & (scan_z_tmp > lower_bound) & (
            scan_z_tmp < upper_bound)]
    depth = depth[(depth > 0) & (depth < max_range) & (scan_z_tmp > lower_bound) & (scan_z_tmp < upper_bound)]
  else:
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]
    depth = depth[(depth > 0) & (depth < max_range)]

  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]

  if scan_z.shape[0] == 0:
    return np.full((proj_H, proj_W), 0, dtype=np.float32)

  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)

  scan_r = depth * np.cos(pitch)

  proj_x = 0.5 * (yaw / np.pi + 1.0)
  proj_y = scan_r / max_range

  proj_x = proj_x * proj_W
  proj_y = proj_y * proj_H

  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)

  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)

  order = np.argsort(scan_z)
  scan_z = scan_z[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  kitti_lidar_height = 2

  proj_bev = np.full((proj_H, proj_W), 0,
                     dtype=np.float32)

  proj_bev[proj_y, proj_x] = scan_z + abs(kitti_lidar_height)

  return proj_bev


def load_vertex(scan_path):
  current_vertex = np.fromfile(scan_path, dtype=np.float32)
  current_vertex = current_vertex.reshape((-1, 4))
  current_points = current_vertex[:, 0:3]
  current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
  current_vertex[:, :-1] = current_points
  return current_vertex


def load_files(folder):
  file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(folder)) for f in fn]
  file_paths.sort()
  return file_paths
