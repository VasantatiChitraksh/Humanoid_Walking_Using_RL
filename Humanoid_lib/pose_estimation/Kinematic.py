import numpy as np

BODY_KEYPOINT_INDICES = {
    "nose": 0,
    "neck": 1,
    "r_shoulder": 2,
    "r_elbow": 3,
    "r_wrist": 4,
    "l_shoulder": 5,
    "l_elbow": 6,
    "l_wrist": 7,
    "mid_hip": 8,
    "r_hip": 9,
    "r_knee": 10,
    "r_ankle": 11,
    "l_hip": 12,
    "l_knee": 13,
    "l_ankle": 14
}


def unit(input_vector):
    vector_magnitude = np.linalg.norm(input_vector)
    return input_vector / (vector_magnitude + 1e-6)


def angle_between(vector_a, vector_b):
    unit_vector_a = unit(vector_a)
    unit_vector_b = unit(vector_b)
    dot_product = np.clip(np.dot(unit_vector_a, unit_vector_b), -1, 1)
    return np.arccos(dot_product)


def compute_spine_angles(neck_position, hip_position):
    spine_vector = unit(neck_position - hip_position)
    spine_yaw = np.arctan2(spine_vector[0], spine_vector[2])
    spine_pitch = -np.arctan2(spine_vector[1], spine_vector[2])
    spine_roll = np.arctan2(spine_vector[1], spine_vector[0])
    return spine_yaw, spine_pitch, spine_roll


def leg_angles(hip_position, knee_position, ankle_position):
    thigh_vector = knee_position - hip_position
    shin_vector = ankle_position - knee_position
    unit_thigh_vector = unit(thigh_vector)

    hip_roll = np.arctan2(unit_thigh_vector[1], unit_thigh_vector[0])
    hip_yaw = np.arctan2(unit_thigh_vector[0], unit_thigh_vector[2])
    hip_pitch = np.arctan2(-unit_thigh_vector[1], unit_thigh_vector[2])

    knee_angle_value = angle_between(thigh_vector, shin_vector)
    return hip_roll, hip_yaw, hip_pitch, knee_angle_value


def arm_angles(shoulder_position, elbow_position, wrist_position):
    upper_arm_vector = elbow_position - shoulder_position
    forearm_vector = wrist_position - elbow_position
    unit_upper_arm_vector = unit(upper_arm_vector)

    shoulder_pitch = np.arctan2(
        -unit_upper_arm_vector[1], unit_upper_arm_vector[2])
    shoulder_roll = np.arctan2(
        unit_upper_arm_vector[0], unit_upper_arm_vector[2])
    elbow_angle_value = angle_between(upper_arm_vector, forearm_vector)
    return shoulder_pitch, shoulder_roll, elbow_angle_value


def body25_to_humanoid_pose(raw_body25_data):
    coordinates_3d = raw_body25_data[:, :3]

    spine_yaw, spine_pitch, spine_roll = compute_spine_angles(
        coordinates_3d[BODY_KEYPOINT_INDICES["neck"]],
        coordinates_3d[BODY_KEYPOINT_INDICES["mid_hip"]]
    )

    right_hip_roll, right_hip_yaw, right_hip_pitch, right_knee_angle = leg_angles(
        coordinates_3d[BODY_KEYPOINT_INDICES["r_hip"]],
        coordinates_3d[BODY_KEYPOINT_INDICES["r_knee"]],
        coordinates_3d[BODY_KEYPOINT_INDICES["r_ankle"]]
    )

    left_hip_roll, left_hip_yaw, left_hip_pitch, left_knee_angle = leg_angles(
        coordinates_3d[BODY_KEYPOINT_INDICES["l_hip"]],
        coordinates_3d[BODY_KEYPOINT_INDICES["l_knee"]],
        coordinates_3d[BODY_KEYPOINT_INDICES["l_ankle"]]
    )

    right_shoulder_pitch, right_shoulder_roll, right_elbow_angle = arm_angles(
        coordinates_3d[BODY_KEYPOINT_INDICES["r_shoulder"]],
        coordinates_3d[BODY_KEYPOINT_INDICES["r_elbow"]],
        coordinates_3d[BODY_KEYPOINT_INDICES["r_wrist"]]
    )

    left_shoulder_pitch, left_shoulder_roll, left_elbow_angle = arm_angles(
        coordinates_3d[BODY_KEYPOINT_INDICES["l_shoulder"]],
        coordinates_3d[BODY_KEYPOINT_INDICES["l_elbow"]],
        coordinates_3d[BODY_KEYPOINT_INDICES["l_wrist"]]
    )

    return np.array([
        spine_yaw, spine_pitch, spine_roll,
        right_hip_roll, right_hip_yaw, right_hip_pitch, right_knee_angle,
        left_hip_roll, left_hip_yaw, left_hip_pitch, left_knee_angle,
        right_shoulder_pitch, right_shoulder_roll, right_elbow_angle,
        left_shoulder_pitch, left_shoulder_roll, left_elbow_angle
    ], dtype=np.float32)
