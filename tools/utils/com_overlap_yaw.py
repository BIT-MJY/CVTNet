from tools.utils.utils import *

def com_overlap_yaw(scan_paths, poses, frame_idx, leg_output_width=360):

  print('Start to compute ground truth overlap and yaw ...')
  overlaps = []
  yaw_idxs = []
  yaw_resolution = leg_output_width
  current_points = load_vertex(scan_paths[frame_idx])
  current_range, project_points, _, _ = range_projection(current_points)
  visible_points = project_points[current_range > 0]
  valid_num = len(visible_points)
  current_pose = poses[frame_idx]

  for reference_idx in range(len(scan_paths)):
    reference_pose = poses[reference_idx]
    reference_points = load_vertex(scan_paths[reference_idx])
    reference_points_world = reference_pose.dot(reference_points.T).T
    reference_points_in_current = np.linalg.inv(current_pose).dot(reference_points_world.T).T
    reference_range, _, _, _ = range_projection(reference_points_in_current)

    overlap = np.count_nonzero(
      abs(reference_range[reference_range > 0] - current_range[reference_range > 0]) < 1) / valid_num
    overlaps.append(overlap)

    # calculate yaw angle
    relative_transform = np.linalg.inv(current_pose).dot(reference_pose)
    relative_rotation = relative_transform[:3, :3]
    _, _, yaw = euler_angles_from_rotation_matrix(relative_rotation)

    yaw_element_idx = int(- (yaw / np.pi) * yaw_resolution // 2 + yaw_resolution // 2)
    yaw_idxs.append(yaw_element_idx)

  ground_truth_mapping = np.zeros((len(scan_paths), 4))
  ground_truth_mapping[:, 0] = np.ones(len(scan_paths)) * frame_idx
  ground_truth_mapping[:, 1] = np.arange(len(scan_paths))
  ground_truth_mapping[:, 2] = overlaps
  ground_truth_mapping[:, 3] = yaw_idxs

  print('Finish generating ground_truth_mapping!')

  return ground_truth_mapping