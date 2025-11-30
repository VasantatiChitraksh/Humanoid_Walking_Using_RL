import cv2
import numpy as np
import mediapipe as mp


class PoseExtractor:
    def __init__(self):
        self.mp_pose_solution = mp.solutions.pose
        self.pose_detector = self.mp_pose_solution.Pose(static_image_mode=False,
                                                        model_complexity=2,
                                                        enable_segmentation=False,
                                                        min_detection_confidence=0.4)

    def extract_keypoints(self, input_image):

        pose_detection_results = self.pose_detector.process(
            (input_image * 255).astype(np.uint8))

        if not pose_detection_results.pose_landmarks:
            return []

        detected_landmarks = pose_detection_results.pose_landmarks.landmark
        all_mediapipe_landmarks = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in detected_landmarks])

        def calculate_midpoint(index_a, index_b):
            point_a, point_b = all_mediapipe_landmarks[index_a], all_mediapipe_landmarks[index_b]
            return (point_a + point_b) / 2.0

        body25_keypoints_list = [
            all_mediapipe_landmarks[0],
            calculate_midpoint(11, 12),
            all_mediapipe_landmarks[12],
            all_mediapipe_landmarks[14],
            all_mediapipe_landmarks[16],
            all_mediapipe_landmarks[11],
            all_mediapipe_landmarks[13],
            all_mediapipe_landmarks[15],
            calculate_midpoint(23, 24),
            all_mediapipe_landmarks[24],
            all_mediapipe_landmarks[26],
            all_mediapipe_landmarks[28],
            all_mediapipe_landmarks[23],
            all_mediapipe_landmarks[25],
            all_mediapipe_landmarks[27],
            all_mediapipe_landmarks[2],
            all_mediapipe_landmarks[5],
            all_mediapipe_landmarks[8],
            all_mediapipe_landmarks[7],
            all_mediapipe_landmarks[32],
            all_mediapipe_landmarks[31],
            all_mediapipe_landmarks[29],
            all_mediapipe_landmarks[28],
            all_mediapipe_landmarks[27],
            all_mediapipe_landmarks[30]
        ]

        final_body25_array = np.array(body25_keypoints_list)
        return final_body25_array

    def draw_skeleton(self, input_image_rgb, body25_keypoints, output_file_path):
        visualized_image_bgr = cv2.cvtColor(
            (input_image_rgb.copy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = visualized_image_bgr.shape

        skeleton_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10), (10, 11),
            (8, 12), (12, 13), (13, 14),
            (0, 15), (15, 17),
            (0, 16), (16, 18),
            (1, 8)
        ]

        for start_joint_index, end_joint_index in skeleton_connections:
            x_start, y_start, z_start, confidence_start = body25_keypoints[start_joint_index]
            x_end, y_end, z_end, confidence_end = body25_keypoints[end_joint_index]
            if confidence_start > 0 and confidence_end > 0:
                cv2.line(visualized_image_bgr, (int(x_start * image_width), int(y_start * image_height)),
                         (int(x_end * image_width), int(y_end * image_height)), (255, 0, 0), 2)

        for joint_index, (x_coord, y_coord, z_coord, visibility_score) in enumerate(body25_keypoints):
            if visibility_score > 0:
                cv2.circle(visualized_image_bgr, (int(x_coord * image_width),
                           int(y_coord * image_height)), 5, (0, 255, 0), -1)
                cv2.putText(visualized_image_bgr, str(joint_index), (int(x_coord * image_width) + 4, int(y_coord * image_height) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if output_file_path:
            cv2.imwrite(output_file_path, visualized_image_bgr)
            print(f"Skeleton overlay saved to: {output_file_path}")
        else:
            cv2.imshow("Skeleton Overlay", visualized_image_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
