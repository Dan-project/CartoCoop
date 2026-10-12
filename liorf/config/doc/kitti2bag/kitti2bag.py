
# import sys

# try:
#     import pykitti
# except ImportError:
#     print("Could not load module 'pykitti'. Please run `pip install pykitti`")
#     sys.exit(1)

# import os
# import cv2
# import numpy as np
# import argparse
# from datetime import datetime
# from tqdm import tqdm

# # ROS 2 imports
# import rclpy
# from rclpy.serialization import serialize_message
# import rosbag2_py

# # ROS 2 message types
# from std_msgs.msg import Header
# from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix, Image
# from sensor_msgs_py import point_cloud2 as pcl2
# from geometry_msgs.msg import TransformStamped, TwistStamped, Transform
# from tf2_msgs.msg import TFMessage
# from builtin_interfaces.msg import Time
# from cv_bridge import CvBridge

# # scipy for quaternion (replaces tf.transformations)
# from scipy.spatial.transform import Rotation as R


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def to_ros_time(seconds: float) -> Time:
#     """Convert a float timestamp (seconds) to builtin_interfaces/Time."""
#     t = Time()
#     t.sec = int(seconds)
#     t.nanosec = int((seconds - t.sec) * 1e9)
#     return t


# def quaternion_from_euler(roll, pitch, yaw):
#     rot = R.from_euler('xyz', [roll, pitch, yaw])
#     q = rot.as_quat()  # [x, y, z, w]
#     return q


# def quaternion_from_matrix(matrix):
#     rot = R.from_matrix(matrix[:3, :3])
#     q = rot.as_quat()  # [x, y, z, w]
#     return q


# def inv(transform):
#     """Invert a rigid body transformation matrix."""
#     Rm = transform[0:3, 0:3]
#     t = transform[0:3, 3]
#     t_inv = -1 * Rm.T.dot(t)
#     transform_inv = np.eye(4)
#     transform_inv[0:3, 0:3] = Rm.T
#     transform_inv[0:3, 3] = t_inv
#     return transform_inv


# def timestamp_from_datetime(dt: datetime) -> float:
#     return dt.timestamp()


# # ---------------------------------------------------------------------------
# # Bag writer helper
# # ---------------------------------------------------------------------------

# class BagWriter:
#     """Thin wrapper around rosbag2_py SequentialWriter."""

#     # Map Python type name → rmw type string
#     TYPE_MAP = {
#         "Imu":              "sensor_msgs/msg/Imu",
#         "NavSatFix":        "sensor_msgs/msg/NavSatFix",
#         "TwistStamped":     "geometry_msgs/msg/TwistStamped",
#         "TFMessage":        "tf2_msgs/msg/TFMessage",
#         "Image":            "sensor_msgs/msg/Image",
#         "CameraInfo":       "sensor_msgs/msg/CameraInfo",
#         "PointCloud2":      "sensor_msgs/msg/PointCloud2",
#     }

#     def __init__(self, output_path: str):
#         self._writer = rosbag2_py.SequentialWriter()
#         storage_options = rosbag2_py.StorageOptions(uri=output_path, storage_id="sqlite3")
#         converter_options = rosbag2_py.ConverterOptions(
#             input_serialization_format="cdr",
#             output_serialization_format="cdr",
#         )
#         self._writer.open(storage_options, converter_options)
#         self._topics = set()

#     def _ensure_topic(self, topic: str, msg):
#         if topic not in self._topics:
#             type_name = type(msg).__name__
#             ros_type = self.TYPE_MAP.get(type_name)
#             if ros_type is None:
#                 raise ValueError(f"Unknown message type: {type_name}")
#             topic_meta = rosbag2_py.TopicMetadata(
#                 name=topic,
#                 type=ros_type,
#                 serialization_format="cdr",
#             )
#             self._writer.create_topic(topic_meta)
#             self._topics.add(topic)

#     def write(self, topic: str, msg, stamp: float):
#         self._ensure_topic(topic, msg)
#         self._writer.write(topic, serialize_message(msg), int(stamp * 1e9))

#     def close(self):
#         del self._writer  # rosbag2_py closes on destruction


# # ---------------------------------------------------------------------------
# # Export functions
# # ---------------------------------------------------------------------------

# def save_imu_data(bag: BagWriter, kitti, imu_frame_id: str, topic: str):
#     print("Exporting IMU (synced)")
#     for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
#         q = quaternion_from_euler(oxts.packet.roll, oxts.packet.pitch, oxts.packet.yaw)
#         stamp = timestamp_from_datetime(timestamp)

#         imu = Imu()
#         imu.header.frame_id = imu_frame_id
#         imu.header.stamp = to_ros_time(stamp)
#         imu.orientation.x = float(q[0])
#         imu.orientation.y = float(q[1])
#         imu.orientation.z = float(q[2])
#         imu.orientation.w = float(q[3])
#         imu.linear_acceleration.x = oxts.packet.af
#         imu.linear_acceleration.y = oxts.packet.al
#         imu.linear_acceleration.z = oxts.packet.au
#         imu.angular_velocity.x = oxts.packet.wf
#         imu.angular_velocity.y = oxts.packet.wl
#         imu.angular_velocity.z = oxts.packet.wu
#         bag.write(topic, imu, stamp)


# def save_imu_data_raw(bag: BagWriter, kitti, imu_frame_id: str, topic: str):
#     print("Exporting IMU Raw (unsynced)")
#     synced_path = kitti.data_path
#     unsynced_path = synced_path.replace('sync', 'extract')
#     imu_path = os.path.join(unsynced_path, 'oxts')

#     # Read timestamps
#     with open(os.path.join(imu_path, 'timestamps.txt')) as f:
#         lines = f.readlines()
#     imu_datetimes = []
#     for line in lines:
#         if len(line) == 1:
#             continue
#         ts = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
#         imu_datetimes.append(ts.timestamp())

#     # Linear time correction
#     imu_index = np.arange(len(imu_datetimes), dtype=np.float64)
#     z = np.polyfit(imu_index, imu_datetimes, 1)
#     imu_datetimes = (z[0] * imu_index + z[1]).tolist()

#     # Read data files
#     imu_data_dir = os.path.join(imu_path, 'data')
#     imu_filenames = sorted(os.listdir(imu_data_dir))
#     imu_data = []
#     for imu_file in imu_filenames:
#         with open(os.path.join(imu_data_dir, imu_file)) as f:
#             for line in f:
#                 if len(line) > 1:
#                     imu_data.append(line.strip().split())

#     assert len(imu_datetimes) == len(imu_data)

#     for stamp, data in zip(imu_datetimes, imu_data):
#         roll, pitch, yaw = float(data[3]), float(data[4]), float(data[5])
#         q = quaternion_from_euler(roll, pitch, yaw)

#         imu = Imu()
#         imu.header.frame_id = imu_frame_id
#         imu.header.stamp = to_ros_time(stamp)
#         imu.orientation.x = float(q[0])
#         imu.orientation.y = float(q[1])
#         imu.orientation.z = float(q[2])
#         imu.orientation.w = float(q[3])
#         imu.linear_acceleration.x = float(data[11])
#         imu.linear_acceleration.y = float(data[12])
#         imu.linear_acceleration.z = float(data[13])
#         imu.angular_velocity.x = float(data[17])
#         imu.angular_velocity.y = float(data[18])
#         imu.angular_velocity.z = float(data[19])
#         bag.write(topic, imu, stamp)

#         # LIO-SAM GPS topic (imu_enu_link frame)
#         imu.header.frame_id = 'imu_enu_link'
#         bag.write('/imu_correct', imu, stamp)


# def save_dynamic_tf(bag: BagWriter, kitti, kitti_type: str, initial_time):
#     print("Exporting time-dependent transformations")
#     if "raw" in kitti_type:
#         for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
#             stamp = timestamp_from_datetime(timestamp)
#             transform = oxts.T_w_imu
#             t = transform[0:3, 3]
#             q = quaternion_from_matrix(transform)

#             tf_stamped = TransformStamped()
#             tf_stamped.header.stamp = to_ros_time(stamp)
#             tf_stamped.header.frame_id = 'world'
#             tf_stamped.child_frame_id = 'odom' # base_link
#             tf_stamped.transform.translation.x = float(t[0])
#             tf_stamped.transform.translation.y = float(t[1])
#             tf_stamped.transform.translation.z = float(t[2])
#             tf_stamped.transform.rotation.x = float(q[0])
#             tf_stamped.transform.rotation.y = float(q[1])
#             tf_stamped.transform.rotation.z = float(q[2])
#             tf_stamped.transform.rotation.w = float(q[3])

#             tf_msg = TFMessage()
#             tf_msg.transforms.append(tf_stamped)
#             bag.write('/tf', tf_msg, stamp)

#     elif "odom" in kitti_type:
#         timestamps = [initial_time + x.total_seconds() for x in kitti.timestamps]
#         for stamp, tf_matrix in zip(timestamps, kitti.T_w_cam0):
#             t = tf_matrix[0:3, 3]
#             q = quaternion_from_matrix(tf_matrix)

#             tf_stamped = TransformStamped()
#             tf_stamped.header.stamp = to_ros_time(stamp)
#             tf_stamped.header.frame_id = 'world'
#             tf_stamped.child_frame_id = 'camera_left'
#             tf_stamped.transform.translation.x = float(t[0])
#             tf_stamped.transform.translation.y = float(t[1])
#             tf_stamped.transform.translation.z = float(t[2])
#             tf_stamped.transform.rotation.x = float(q[0])
#             tf_stamped.transform.rotation.y = float(q[1])
#             tf_stamped.transform.rotation.z = float(q[2])
#             tf_stamped.transform.rotation.w = float(q[3])

#             tf_msg = TFMessage()
#             tf_msg.transforms.append(tf_stamped)
#             bag.write('/tf', tf_msg, stamp)


# def save_camera_data(bag: BagWriter, kitti_type: str, kitti, util, bridge: CvBridge,
#                      camera: int, camera_frame_id: str, topic: str, initial_time):
#     print(f"Exporting camera {camera}")
#     if "raw" in kitti_type:
#         camera_pad = f'{camera:02d}'
#         image_dir = os.path.join(kitti.data_path, f'image_{camera_pad}')
#         image_path = os.path.join(image_dir, 'data')
#         image_filenames = sorted(os.listdir(image_path))
#         with open(os.path.join(image_dir, 'timestamps.txt')) as f:
#             image_datetimes = [
#                 datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
#                 for line in f if len(line) > 1
#             ]

#         calib = CameraInfo()
#         calib.header.frame_id = camera_frame_id
#         calib.width, calib.height = [int(v) for v in util[f'S_rect_{camera_pad}'].tolist()]
#         calib.distortion_model = 'plumb_bob'
#         calib.k = util[f'K_{camera_pad}'].flatten().tolist()
#         calib.r = util[f'R_rect_{camera_pad}'].flatten().tolist()
#         calib.d = util[f'D_{camera_pad}'].flatten().tolist()
#         calib.p = util[f'P_rect_{camera_pad}'].flatten().tolist()
#         topic_ext = "/image_raw"

#     elif "odom" in kitti_type:
#         camera_pad = f'{camera:01d}'
#         image_path = os.path.join(kitti.sequence_path, f'image_{camera_pad}')
#         image_filenames = sorted(os.listdir(image_path))
#         image_datetimes = [initial_time + x.total_seconds() for x in kitti.timestamps]

#         calib = CameraInfo()
#         calib.header.frame_id = camera_frame_id
#         calib.p = util[f'P{camera_pad}'].flatten().tolist()
#         topic_ext = "/image_rect"

#     for dt, filename in tqdm(zip(image_datetimes, image_filenames), total=len(image_filenames)):
#         image_filename = os.path.join(image_path, filename)
#         cv_image = cv2.imread(image_filename)
#         if cv_image is None:
#             continue
#         calib.height, calib.width = cv_image.shape[:2]

#         if camera in (0, 1):
#             cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
#         encoding = "mono8" if camera in (0, 1) else "bgr8"

#         image_message = bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
#         image_message.header.frame_id = camera_frame_id

#         if "raw" in kitti_type:
#             stamp = timestamp_from_datetime(dt)
#         else:
#             stamp = dt

#         image_message.header.stamp = to_ros_time(stamp)
#         calib.header.stamp = image_message.header.stamp

#         bag.write(topic + topic_ext, image_message, stamp)
#         bag.write(topic + '/camera_info', calib, stamp)


# def save_velo_data(bag: BagWriter, kitti, velo_frame_id: str, topic: str):
#     print("Exporting Velodyne data")
#     velo_path = os.path.join(kitti.data_path, 'velodyne_points')
#     velo_data_dir = os.path.join(velo_path, 'data')
#     velo_filenames = sorted(os.listdir(velo_data_dir))

#     with open(os.path.join(velo_path, 'timestamps.txt')) as f:
#         lines = f.readlines()
#     velo_datetimes = []
#     for line in lines:
#         if len(line) == 1:
#             continue
#         dt = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
#         velo_datetimes.append(dt)

#     for dt, filename in tqdm(zip(velo_datetimes, velo_filenames), total=len(velo_filenames)):
#         if dt is None:
#             continue
#         stamp = timestamp_from_datetime(dt)
#         velo_filename = os.path.join(velo_data_dir, filename)

#         scan = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4)

#         # Compute ring channel
#         depth = np.linalg.norm(scan, 2, axis=1)
#         pitch = np.arcsin(scan[:, 2] / depth)
#         fov_down = -24.8 / 180.0 * np.pi
#         fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
#         proj_y = (pitch + abs(fov_down)) / fov
#         proj_y = np.clip(np.floor(proj_y * 64), 0, 63).astype(np.int32).reshape(-1, 1)
#         scan = np.concatenate((scan, proj_y), axis=1).tolist()
#         for i in range(len(scan)):
#             scan[i][-1] = int(scan[i][-1])

#         header = Header()
#         header.frame_id = velo_frame_id
#         header.stamp = to_ros_time(stamp)

#         fields = [
#             PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
#             PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
#             PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
#             PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
#             PointField(name='ring',      offset=16, datatype=PointField.UINT16,  count=1),
#         ]
#         pcl_msg = pcl2.create_cloud(header, fields, scan)
#         pcl_msg.is_dense = True
#         bag.write(topic, pcl_msg, stamp)


# def get_static_transform(from_frame_id: str, to_frame_id: str, transform) -> TransformStamped:
#     t = transform[0:3, 3]
#     q = quaternion_from_matrix(transform)
#     tf_msg = TransformStamped()
#     tf_msg.header.frame_id = from_frame_id
#     tf_msg.child_frame_id = to_frame_id
#     tf_msg.transform.translation.x = float(t[0])
#     tf_msg.transform.translation.y = float(t[1])
#     tf_msg.transform.translation.z = float(t[2])
#     tf_msg.transform.rotation.x = float(q[0])
#     tf_msg.transform.rotation.y = float(q[1])
#     tf_msg.transform.rotation.z = float(q[2])
#     tf_msg.transform.rotation.w = float(q[3])
#     return tf_msg


# def save_static_transforms(bag: BagWriter, transforms, timestamps):
#     print("Exporting static transformations")
#     tfm = TFMessage()
#     for from_id, to_id, mat in transforms:
#         tfm.transforms.append(get_static_transform(from_id, to_id, mat))

#     for timestamp in timestamps:
#         stamp = timestamp_from_datetime(timestamp)
#         for tf in tfm.transforms:
#             tf.header.stamp = to_ros_time(stamp)
#         bag.write('/tf_static', tfm, stamp)


# def save_gps_fix_data(bag: BagWriter, kitti, gps_frame_id: str, topic: str):
#     print("Exporting GPS fix")
#     for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
#         stamp = timestamp_from_datetime(timestamp)
#         msg = NavSatFix()
#         msg.header.frame_id = gps_frame_id
#         msg.header.stamp = to_ros_time(stamp)
#         msg.latitude = oxts.packet.lat
#         msg.longitude = oxts.packet.lon
#         msg.altitude = oxts.packet.alt
#         msg.status.service = 1
#         # rajout pour ne pas avoir 0 
#         msg.position_covariance[0] = oxts.packet.pos_accuracy ** 2 # x
#         msg.position_covariance[4] = oxts.packet.pos_accuracy ** 2  # y
#         msg.position_covariance[8] = oxts.packet.pos_accuracy ** 2   # z
#         msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
#         bag.write(topic, msg, stamp)


# def save_gps_vel_data(bag: BagWriter, kitti, gps_frame_id: str, topic: str):
#     print("Exporting GPS velocity")
#     for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
#         stamp = timestamp_from_datetime(timestamp)
#         msg = TwistStamped()
#         msg.header.frame_id = gps_frame_id
#         msg.header.stamp = to_ros_time(stamp)
#         msg.twist.linear.x = oxts.packet.vf
#         msg.twist.linear.y = oxts.packet.vl
#         msg.twist.linear.z = oxts.packet.vu
#         msg.twist.angular.x = oxts.packet.wf
#         msg.twist.angular.y = oxts.packet.wl
#         msg.twist.angular.z = oxts.packet.wu
#         bag.write(topic, msg, stamp)


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(description="Convert KITTI dataset to ROS 2 bag file")
#     kitti_types = ["raw_synced", "odom_color", "odom_gray"]
#     odometry_sequences = [str(s).zfill(2) for s in range(22)]

#     parser.add_argument("kitti_type", choices=kitti_types, help="KITTI dataset type")
#     parser.add_argument("dir", nargs="?", default=os.getcwd(),
#                         help="Base directory of the dataset (default: cwd)")
#     parser.add_argument("-t", "--date", help="Date of the raw dataset (e.g. 2011_09_26). Raw only.")
#     parser.add_argument("-r", "--drive", help="Drive number (e.g. 0001). Raw only.")
#     parser.add_argument("-s", "--sequence", choices=odometry_sequences,
#                         help="Sequence number 00-21. Odometry only.")
#     args = parser.parse_args()

#     rclpy.init()
#     bridge = CvBridge()

#     cameras = [
#         (0, 'camera_gray_left',  '/kitti/camera_gray_left'),
#         (1, 'camera_gray_right', '/kitti/camera_gray_right'),
#         (2, 'camera_color_left', '/kitti/camera_color_left'),
#         (3, 'camera_color_right','/kitti/camera_color_right'),
#     ]

#     # ------------------------------------------------------------------
#     # RAW dataset
#     # ------------------------------------------------------------------
#     if "raw" in args.kitti_type:
#         if not args.date or not args.drive:
#             print("--date and --drive are required for raw datasets.")
#             print("Usage: kitti2bag_ros2.py raw_synced [dir] -t <date> -r <drive>")
#             sys.exit(1)

#         bag_path = f"kitti_{args.date}_drive_{args.drive}_{args.kitti_type[4:]}"
#         bag = BagWriter(bag_path)
#         kitti = pykitti.raw(args.dir, args.date, args.drive)

#         if not os.path.exists(kitti.data_path):
#             print(f"Path {kitti.data_path} does not exist. Exiting.")
#             sys.exit(1)
#         if len(kitti.timestamps) == 0:
#             print("Dataset is empty. Exiting.")
#             sys.exit(1)

#         try:
#             imu_frame_id  = 'imu_link'
#             imu_topic     = '/kitti/oxts/imu'
#             imu_raw_topic = '/imu_raw'
#             gps_fix_topic = '/gps/fix'
#             gps_vel_topic = '/gps/vel'
#             velo_frame_id = 'lidar_link'
#             velo_topic    = '/points_raw'

#             T_base_link_to_imu = np.eye(4)
#             T_base_link_to_imu[0:3, 3] = [-2.71 / 2.0 - 0.05, 0.32, 0.93]

#             transforms = [
#                 ('base_link',  imu_frame_id,    T_base_link_to_imu),
#                 (imu_frame_id, velo_frame_id,   inv(kitti.calib.T_velo_imu)),
#                 (imu_frame_id, cameras[0][1],   inv(kitti.calib.T_cam0_imu)),
#                 (imu_frame_id, cameras[1][1],   inv(kitti.calib.T_cam1_imu)),
#                 (imu_frame_id, cameras[2][1],   inv(kitti.calib.T_cam2_imu)),
#                 (imu_frame_id, cameras[3][1],   inv(kitti.calib.T_cam3_imu)),
#             ]

#             util = pykitti.utils.read_calib_file(
#                 os.path.join(kitti.calib_path, 'calib_cam_to_cam.txt')
#             )

#             # Uncomment to enable static TF and dynamic TF export:
#             save_static_transforms(bag, transforms, kitti.timestamps)
#             save_dynamic_tf(bag, kitti, args.kitti_type, initial_time=None)
#             save_imu_data(bag, kitti, imu_frame_id, imu_topic)

#             save_imu_data_raw(bag, kitti, imu_frame_id, imu_raw_topic)
#             save_gps_fix_data(bag, kitti, imu_frame_id, gps_fix_topic)
#             save_gps_vel_data(bag, kitti, imu_frame_id, gps_vel_topic)

#             # Export only camera 0 (add more by removing the `break`)
#             for camera in cameras:
#                 save_camera_data(bag, args.kitti_type, kitti, util, bridge,
#                                  camera=camera[0], camera_frame_id=camera[1],
#                                  topic=camera[2], initial_time=None)
#                 # break
 
#             save_velo_data(bag, kitti, velo_frame_id, velo_topic)

#         finally:
#             bag.close()
#             print(f"Bag written to: {bag_path}/")

#     # ------------------------------------------------------------------
#     # ODOMETRY dataset
#     # ------------------------------------------------------------------
#     elif "odom" in args.kitti_type:
#         if not args.sequence:
#             print("--sequence is required for odometry datasets.")
#             print("Usage: kitti2bag_ros2.py {odom_color,odom_gray} [dir] -s <sequence>")
#             sys.exit(1)

#         bag_path = f"kitti_data_odometry_{args.kitti_type[5:]}_sequence_{args.sequence}"
#         bag = BagWriter(bag_path)
#         kitti = pykitti.odometry(args.dir, args.sequence)

#         if not os.path.exists(kitti.sequence_path):
#             print(f"Path {kitti.sequence_path} does not exist. Exiting.")
#             sys.exit(1)

#         kitti.load_calib()
#         kitti.load_timestamps()

#         if len(kitti.timestamps) == 0:
#             print("Dataset is empty. Exiting.")
#             sys.exit(1)

#         if args.sequence in odometry_sequences[:11]:
#             print(f"Sequence {args.sequence} has ground truth poses.")
#             kitti.load_poses()

#         try:
#             util = pykitti.utils.read_calib_file(
#                 os.path.join(args.dir, 'sequences', args.sequence, 'calib.txt')
#             )
#             current_epoch = (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()

#             used_cameras = cameras[:2] if "gray" in args.kitti_type else cameras[-2:]

#             save_dynamic_tf(bag, kitti, args.kitti_type, initial_time=current_epoch)
#             for camera in used_cameras:
#                 save_camera_data(bag, args.kitti_type, kitti, util, bridge,
#                                  camera=camera[0], camera_frame_id=camera[1],
#                                  topic=camera[2], initial_time=current_epoch)
#         finally:
#             bag.close()
#             print(f"Bag written to: {bag_path}/")

#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

try:
    import pykitti
except ImportError:
    print("Could not load module 'pykitti'. Please run `pip install pykitti`")
    sys.exit(1)

import os
import cv2
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

import rclpy
from rclpy.serialization import serialize_message
import rosbag2_py

from scipy.spatial.transform import Rotation

from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix, PointCloud2
import sensor_msgs_py.point_cloud2 as pcl2
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time as RosTime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_ros_time(seconds: float) -> RosTime:
    t = RosTime()
    t.sec = int(seconds)
    t.nanosec = int(round((seconds - t.sec) * 1e9))
    return t


def timestamp_to_sec(dt: datetime) -> float:
    return dt.timestamp()


def quaternion_from_euler(roll, pitch, yaw):
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    return r.as_quat()  # [x, y, z, w]


def quaternion_from_matrix(matrix):
    r = Rotation.from_matrix(matrix[:3, :3])
    return r.as_quat()  # [x, y, z, w]


def inv(transform):
    """Invert a rigid body transformation matrix."""
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv
    return transform_inv


def nanosec_from_sec(seconds: float) -> int:
    return int(seconds * 1e9)


def create_writer(bag_path: str) -> rosbag2_py.SequentialWriter:
    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    writer.open(storage_options, converter_options)
    return writer


def add_topic(writer: rosbag2_py.SequentialWriter, topic: str, msg_type: str):
    topic_meta = rosbag2_py.TopicMetadata(
        name=topic,
        type=msg_type,
        serialization_format='cdr'
    )
    writer.create_topic(topic_meta)


def write_msg(writer, topic, msg, stamp_sec: float):
    writer.write(topic, serialize_message(msg), nanosec_from_sec(stamp_sec))


# ---------------------------------------------------------------------------
# IMU (synced)
# ---------------------------------------------------------------------------

def save_imu_data(writer, kitti, imu_frame_id, topic):
    print("Exporting IMU")
    add_topic(writer, topic, 'sensor_msgs/msg/Imu')
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        q = quaternion_from_euler(oxts.packet.roll, oxts.packet.pitch, oxts.packet.yaw)
        imu = Imu()
        imu.header.frame_id = imu_frame_id
        t_sec = timestamp_to_sec(timestamp)
        imu.header.stamp = to_ros_time(t_sec)
        imu.orientation.x = q[0]
        imu.orientation.y = q[1]
        imu.orientation.z = q[2]
        imu.orientation.w = q[3]
        imu.linear_acceleration.x = oxts.packet.af
        imu.linear_acceleration.y = oxts.packet.al
        imu.linear_acceleration.z = oxts.packet.au
        imu.angular_velocity.x = oxts.packet.wf
        imu.angular_velocity.y = oxts.packet.wl
        imu.angular_velocity.z = oxts.packet.wu
        write_msg(writer, topic, imu, t_sec)


# ---------------------------------------------------------------------------
# IMU (raw / unsynced) + GPS @ 100 Hz
# ---------------------------------------------------------------------------

def save_imu_data_raw(writer, kitti, imu_frame_id, topic, gps_correct_topic):
    print("Exporting IMU Raw")
    add_topic(writer, topic, 'sensor_msgs/msg/Imu')
    add_topic(writer, '/imu_correct', 'sensor_msgs/msg/Imu')
    add_topic(writer, gps_correct_topic, 'sensor_msgs/msg/NavSatFix')

    synced_path = kitti.data_path
    unsynced_path = synced_path.replace('sync', 'extract')
    imu_path = os.path.join(unsynced_path, 'oxts')

    with open(os.path.join(imu_path, 'timestamps.txt')) as f:
        lines = f.readlines()
    imu_datetimes = []
    for line in lines:
        if len(line) == 1:
            continue
        timestamp = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
        imu_datetimes.append(timestamp.timestamp())

    # linear model correction
    imu_index = np.arange(len(imu_datetimes), dtype=np.float64)
    z = np.polyfit(imu_index, imu_datetimes, 1)
    imu_datetimes = (z[0] * imu_index + z[1]).tolist()

    imu_data_dir = os.path.join(imu_path, 'data')
    imu_filenames = sorted(os.listdir(imu_data_dir))
    imu_data = []
    for imu_file in imu_filenames:
        with open(os.path.join(imu_data_dir, imu_file)) as f:
            for line in f:
                if len(line) > 1:
                    imu_data.append(line.strip().split())

    assert len(imu_datetimes) == len(imu_data)

    for t_sec, data in zip(imu_datetimes, imu_data):
        roll, pitch, yaw = float(data[3]), float(data[4]), float(data[5])
        q = quaternion_from_euler(roll, pitch, yaw)

        imu = Imu()
        imu.header.frame_id = imu_frame_id
        imu.header.stamp = to_ros_time(t_sec)
        imu.orientation.x, imu.orientation.y = q[0], q[1]
        imu.orientation.z, imu.orientation.w = q[2], q[3]
        imu.linear_acceleration.x = float(data[11])
        imu.linear_acceleration.y = float(data[12])
        imu.linear_acceleration.z = float(data[13])
        imu.angular_velocity.x = float(data[17])
        imu.angular_velocity.y = float(data[18])
        imu.angular_velocity.z = float(data[19])
        write_msg(writer, topic, imu, t_sec)

        imu.header.frame_id = 'imu_enu_link'
        write_msg(writer, '/imu_correct', imu, t_sec)

        nav = NavSatFix()
        nav.header.frame_id = imu_frame_id
        nav.header.stamp = to_ros_time(t_sec)
        nav.latitude = float(data[0])
        nav.longitude = float(data[1])
        nav.altitude = float(data[2])
        nav.status.service = 1
        write_msg(writer, gps_correct_topic, nav, t_sec)


# ---------------------------------------------------------------------------
# Dynamic TF
# ---------------------------------------------------------------------------

def save_dynamic_tf(writer, kitti, kitti_type, initial_time):
    print("Exporting time-dependent transformations")
    add_topic(writer, '/tf', 'tf2_msgs/msg/TFMessage')

    if 'raw' in kitti_type:
        for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
            t_sec = timestamp_to_sec(timestamp)
            tf_stamped = TransformStamped()
            tf_stamped.header.stamp = to_ros_time(t_sec)
            tf_stamped.header.frame_id = 'world'
            tf_stamped.child_frame_id = 'base_link'

            T = oxts.T_w_imu
            tr = T[0:3, 3]
            q = quaternion_from_matrix(T)
            tf_stamped.transform.translation.x = float(tr[0])
            tf_stamped.transform.translation.y = float(tr[1])
            tf_stamped.transform.translation.z = float(tr[2])
            tf_stamped.transform.rotation.x = float(q[0])
            tf_stamped.transform.rotation.y = float(q[1])
            tf_stamped.transform.rotation.z = float(q[2])
            tf_stamped.transform.rotation.w = float(q[3])

            tfm = TFMessage(transforms=[tf_stamped])
            write_msg(writer, '/tf', tfm, t_sec)

    elif 'odom' in kitti_type:
        timestamps = [initial_time + x.total_seconds() for x in kitti.timestamps]
        for t_sec, tf_matrix in zip(timestamps, kitti.T_w_cam0):
            tf_stamped = TransformStamped()
            tf_stamped.header.stamp = to_ros_time(t_sec)
            tf_stamped.header.frame_id = 'world'
            tf_stamped.child_frame_id = 'camera_left'

            tr = tf_matrix[0:3, 3]
            q = quaternion_from_matrix(tf_matrix)
            tf_stamped.transform.translation.x = float(tr[0])
            tf_stamped.transform.translation.y = float(tr[1])
            tf_stamped.transform.translation.z = float(tr[2])
            tf_stamped.transform.rotation.x = float(q[0])
            tf_stamped.transform.rotation.y = float(q[1])
            tf_stamped.transform.rotation.z = float(q[2])
            tf_stamped.transform.rotation.w = float(q[3])

            tfm = TFMessage(transforms=[tf_stamped])
            write_msg(writer, '/tf', tfm, t_sec)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def save_camera_data(writer, kitti_type, kitti, util, bridge, camera, camera_frame_id, topic, initial_time):
    print("Exporting camera {}".format(camera))

    topic_ext = '/image_raw' if 'raw' in kitti_type else '/image_rect'
    add_topic(writer, topic + topic_ext, 'sensor_msgs/msg/Image')
    add_topic(writer, topic + '/camera_info', 'sensor_msgs/msg/CameraInfo')

    if 'raw' in kitti_type:
        camera_pad = '{0:02d}'.format(camera)
        image_dir = os.path.join(kitti.data_path, 'image_{}'.format(camera_pad))
        image_path = os.path.join(image_dir, 'data')
        image_filenames = sorted(os.listdir(image_path))
        with open(os.path.join(image_dir, 'timestamps.txt')) as f:
            image_datetimes = [
                datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                for line in f if len(line) > 1
            ]

        calib = CameraInfo()
        calib.header.frame_id = camera_frame_id
        calib.width, calib.height = [int(x) for x in util['S_rect_00'].tolist()]  
        calib.distortion_model = 'plumb_bob'
        calib.k = util['K_{}'.format(camera_pad)].flatten().tolist()
        calib.r = util['R_rect_{}'.format(camera_pad)].flatten().tolist()
        calib.d = util['D_{}'.format(camera_pad)].flatten().tolist()
        calib.p = util['P_rect_{}'.format(camera_pad)].flatten().tolist()

    elif 'odom' in kitti_type:
        camera_pad = '{0:01d}'.format(camera)
        image_path = os.path.join(kitti.sequence_path, 'image_{}'.format(camera_pad))
        image_filenames = sorted(os.listdir(image_path))
        image_datetimes = [initial_time + x.total_seconds() for x in kitti.timestamps]

        calib = CameraInfo()
        calib.header.frame_id = camera_frame_id
        calib.p = util['P{}'.format(camera_pad)].flatten().tolist()

    for dt, filename in tqdm(zip(image_datetimes, image_filenames), total=len(image_filenames)):
        image_filename = os.path.join(image_path, filename)
        cv_image = cv2.imread(image_filename)
        calib.height, calib.width = cv_image.shape[:2]

        if camera in (0, 1):
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        encoding = 'mono8' if camera in (0, 1) else 'bgr8'
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
        image_message.header.frame_id = camera_frame_id

        if 'raw' in kitti_type:
            t_sec = timestamp_to_sec(dt)
        else:
            t_sec = dt  # already a float

        image_message.header.stamp = to_ros_time(t_sec)
        calib.header.stamp = image_message.header.stamp

        write_msg(writer, topic + topic_ext, image_message, t_sec)
        write_msg(writer, topic + '/camera_info', calib, t_sec)


# ---------------------------------------------------------------------------
# Velodyne
# ---------------------------------------------------------------------------

def save_velo_data(writer, kitti, velo_frame_id, topic):
    print("Exporting velodyne data")
    add_topic(writer, topic, 'sensor_msgs/msg/PointCloud2')

    velo_path = os.path.join(kitti.data_path, 'velodyne_points')
    velo_data_dir = os.path.join(velo_path, 'data')
    velo_filenames = sorted(os.listdir(velo_data_dir))

    with open(os.path.join(velo_path, 'timestamps.txt')) as f:
        lines = f.readlines()
    velo_datetimes = []
    for line in lines:
        if len(line) == 1:
            continue
        dt = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
        velo_datetimes.append(dt)

    for dt, filename in tqdm(zip(velo_datetimes, velo_filenames), total=len(velo_filenames)):
        if dt is None:
            continue

        velo_filename = os.path.join(velo_data_dir, filename)
        scan = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4)

        depth = np.linalg.norm(scan, 2, axis=1)
        pitch = np.arcsin(scan[:, 2] / depth)
        fov_down = -24.8 / 180.0 * np.pi
        fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
        proj_y = (pitch + abs(fov_down)) / fov
        proj_y = np.clip(np.floor(proj_y * 64), 0, 63).astype(np.int32)

        scan_with_ring = np.concatenate((scan, proj_y.reshape(-1, 1)), axis=1).tolist()
        for i in range(len(scan_with_ring)):
            scan_with_ring[i][-1] = int(scan_with_ring[i][-1])

        t_sec = timestamp_to_sec(dt)
        header = Header()
        header.frame_id = velo_frame_id
        header.stamp = to_ros_time(t_sec)

        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring',      offset=16, datatype=PointField.UINT16,  count=1),
        ]
        pcl_msg = pcl2.create_cloud(header, fields, scan_with_ring)
        pcl_msg.is_dense = True

        write_msg(writer, topic, pcl_msg, t_sec)


# ---------------------------------------------------------------------------
# Static TF
# ---------------------------------------------------------------------------

def get_static_transform(from_frame_id, to_frame_id, transform):
    tr = transform[0:3, 3]
    q = quaternion_from_matrix(transform)
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = from_frame_id
    tf_msg.child_frame_id = to_frame_id
    tf_msg.transform.translation.x = float(tr[0])
    tf_msg.transform.translation.y = float(tr[1])
    tf_msg.transform.translation.z = float(tr[2])
    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])
    return tf_msg


def save_static_transforms(writer, transforms, timestamps):
    print("Exporting static transformations")
    add_topic(writer, '/tf_static', 'tf2_msgs/msg/TFMessage')
    tfm = TFMessage()
    for (from_f, to_f, T) in transforms:
        tfm.transforms.append(get_static_transform(from_f, to_f, T))
    for timestamp in timestamps:
        t_sec = timestamp_to_sec(timestamp)
        for t in tfm.transforms:
            t.header.stamp = to_ros_time(t_sec)
        write_msg(writer, '/tf_static', tfm, t_sec)


# ---------------------------------------------------------------------------
# GPS fix / vel
# ---------------------------------------------------------------------------

def save_gps_fix_data(writer, kitti, gps_frame_id, topic):
    add_topic(writer, topic, 'sensor_msgs/msg/NavSatFix')
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        t_sec = timestamp_to_sec(timestamp)
        nav = NavSatFix()
        nav.header.frame_id = gps_frame_id
        nav.header.stamp = to_ros_time(t_sec)
        nav.latitude = oxts.packet.lat
        nav.longitude = oxts.packet.lon
        nav.altitude = oxts.packet.alt
        nav.status.service = 1
        write_msg(writer, topic, nav, t_sec)


def save_gps_vel_data(writer, kitti, gps_frame_id, topic):
    add_topic(writer, topic, 'geometry_msgs/msg/TwistStamped')
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        t_sec = timestamp_to_sec(timestamp)
        twist = TwistStamped()
        twist.header.frame_id = gps_frame_id
        twist.header.stamp = to_ros_time(t_sec)
        twist.twist.linear.x = oxts.packet.vf
        twist.twist.linear.y = oxts.packet.vl
        twist.twist.linear.z = oxts.packet.vu
        twist.twist.angular.x = oxts.packet.wf
        twist.twist.angular.y = oxts.packet.wl
        twist.twist.angular.z = oxts.packet.wu
        write_msg(writer, topic, twist, t_sec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert KITTI dataset to ROS2 bag file')
    kitti_types = ['raw_synced', 'odom_color', 'odom_gray']
    odometry_sequences = [str(s).zfill(2) for s in range(22)]

    parser.add_argument('kitti_type', choices=kitti_types, help='KITTI dataset type')
    parser.add_argument('dir', nargs='?', default=os.getcwd(), help='Base directory of the dataset')
    parser.add_argument('-t', '--date', help='Date of the raw dataset (e.g. 2011_09_26)')
    parser.add_argument('-r', '--drive', help='Drive number (e.g. 0001)')
    parser.add_argument('-s', '--sequence', choices=odometry_sequences, help='Odometry sequence (00-21)')
    args = parser.parse_args()

    rclpy.init()
    bridge = CvBridge()

    cameras = [
        (0, 'camera_gray_left',  '/kitti/camera_gray_left'),
        (1, 'camera_gray_right', '/kitti/camera_gray_right'),
        (2, 'camera_color_left', '/kitti/camera_color_left'),
        (3, 'camera_color_right','/kitti/camera_color_right'),
    ]

    # ------------------------------------------------------------------
    if 'raw' in args.kitti_type:
        if args.date is None or args.drive is None:
            print("Both --date and --drive are required for raw datasets.")
            print("Usage: kitti2bag_ros2.py raw_synced [dir] -t <date> -r <drive>")
            sys.exit(1)

        bag_path = 'kitti_{}_drive_{}_{}'.format(args.date, args.drive, args.kitti_type[4:])
        writer = create_writer(bag_path)
        kitti = pykitti.raw(args.dir, args.date, args.drive)

        if not os.path.exists(kitti.data_path):
            print('Path {} does not exist. Exiting.'.format(kitti.data_path))
            sys.exit(1)
        if len(kitti.timestamps) == 0:
            print('Dataset is empty? Exiting.')
            sys.exit(1)

        imu_frame_id       = 'imu_link'
        imu_topic          = '/kitti/oxts/imu'
        imu_raw_topic      = '/imu_raw'
        gps_fix_topic      = '/gps/fix'
        gps_fix_correct_topic = '/gps/fix/correct'
        gps_vel_topic      = '/gps/vel'
        velo_frame_id      = 'velodyne'
        velo_topic         = '/points_raw'

        T_base_link_to_imu = np.eye(4)
        T_base_link_to_imu[0:3, 3] = [-2.71 / 2.0 - 0.05, 0.32, 0.93]

        transforms = [
            ('base_link',   imu_frame_id,     T_base_link_to_imu),
            (imu_frame_id,  velo_frame_id,    inv(kitti.calib.T_velo_imu)),
            (imu_frame_id,  cameras[0][1],    inv(kitti.calib.T_cam0_imu)),
            (imu_frame_id,  cameras[1][1],    inv(kitti.calib.T_cam1_imu)),
            (imu_frame_id,  cameras[2][1],    inv(kitti.calib.T_cam2_imu)),
            (imu_frame_id,  cameras[3][1],    inv(kitti.calib.T_cam3_imu)),
        ]

        util = pykitti.utils.read_calib_file(
            os.path.join(kitti.calib_path, 'calib_cam_to_cam.txt'))

        # Uncomment as needed:
        # save_static_transforms(writer, transforms, kitti.timestamps)
        save_dynamic_tf(writer, kitti, args.kitti_type, initial_time=None)
        # save_imu_data(writer, kitti, imu_frame_id, imu_topic)
        save_imu_data_raw(writer, kitti, imu_frame_id, imu_raw_topic, gps_fix_correct_topic)
        save_gps_fix_data(writer, kitti, imu_frame_id, gps_fix_topic)
        save_gps_vel_data(writer, kitti, imu_frame_id, gps_vel_topic)
        for camera in cameras:
            save_camera_data(writer, args.kitti_type, kitti, util, bridge,
                             camera=camera[0], camera_frame_id=camera[1],
                             topic=camera[2], initial_time=None)
            # break  # only first camera remove it for more camera
        save_velo_data(writer, kitti, velo_frame_id, velo_topic)

        print("## Done – bag written to:", bag_path)

    # ------------------------------------------------------------------
    elif 'odom' in args.kitti_type:
        if args.sequence is None:
            print("--sequence is required for odometry datasets.")
            print("Usage: kitti2bag_ros2.py {odom_color,odom_gray} [dir] -s <sequence>")
            sys.exit(1)

        bag_path = 'kitti_data_odometry_{}_sequence_{}'.format(
            args.kitti_type[5:], args.sequence)
        writer = create_writer(bag_path)
        kitti = pykitti.odometry(args.dir, args.sequence)

        if not os.path.exists(kitti.sequence_path):
            print('Path {} does not exist. Exiting.'.format(kitti.sequence_path))
            sys.exit(1)

        kitti.load_calib()
        kitti.load_timestamps()

        if len(kitti.timestamps) == 0:
            print('Dataset is empty? Exiting.')
            sys.exit(1)

        if args.sequence in odometry_sequences[:11]:
            print("Sequence {} has ground truth poses.".format(args.sequence))
            kitti.load_poses()

        util = pykitti.utils.read_calib_file(
            os.path.join(args.dir, 'sequences', args.sequence, 'calib.txt'))
        current_epoch = (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()

        used_cameras = cameras[:2] if 'gray' in args.kitti_type else cameras[-2:]

        save_dynamic_tf(writer, kitti, args.kitti_type, initial_time=current_epoch)
        for camera in used_cameras:
            save_camera_data(writer, args.kitti_type, kitti, util, bridge,
                             camera=camera[0], camera_frame_id=camera[1],
                             topic=camera[2], initial_time=current_epoch)

        print("## Done – bag written to:", bag_path)

    rclpy.shutdown()