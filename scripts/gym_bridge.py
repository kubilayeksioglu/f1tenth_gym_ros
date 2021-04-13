#!/usr/bin/env python
from os import name
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform, Point
from geometry_msgs.msg import Quaternion, Vector3
from ackermann_msgs.msg import AckermannDriveStamped

from f1tenth_gym_ros.msg import RaceInfo
from tf2_ros import transform_broadcaster
from tf.transformations import quaternion_from_euler

import numpy as np

from agents import PurePursuitAgent


def rotation_from_steer(steer):
    rot = Quaternion()
    wheel_quat = quaternion_from_euler(0., 0., steer)
    rot.x, rot.y, rot.z, rot.w = wheel_quat
    return rot


class GymBridge(object):
    def __init__(self):
        # get topic list
        self.ego_scan_topic = rospy.get_param('ego_scan_topic')
        self.ego_odom_topic = rospy.get_param('ego_odom_topic')
        self.opp_odom_topic = rospy.get_param('opp_odom_topic')
        self.ego_drive_topic = rospy.get_param('ego_drive_topic')
        self.race_info_topic = rospy.get_param('race_info_topic')

        # this keeps
        self.scan_distance_to_base_link = rospy.get_param('scan_distance_to_base_link')

        self.map_path = rospy.get_param('map_path')
        self.map_img_ext = rospy.get_param('map_img_ext')
        print(self.map_path, self.map_img_ext)
        exec_dir = rospy.get_param('executable_dir')

        scan_fov = rospy.get_param('scan_fov')
        scan_beams = rospy.get_param('scan_beams')
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams

        csv_path = rospy.get_param('waypoints_path')

        wheelbase = 0.3302
        mass = 3.47
        l_r = 0.17145
        I_z = 0.04712
        mu = 0.523
        h_cg = 0.074
        cs_f = 4.718
        cs_r = 5.4562
        # init gym backend
        self.racecar_env = gym.make('f110_gym:f110-v0')
        self.racecar_env.init_map(self.map_path, self.map_img_ext, False, False)
        self.racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, exec_dir, double_finish=True)

        # init opponent agent
        # TODO: init by params.yaml
        self.opp_agent = PurePursuitAgent(csv_path, wheelbase)
        initial_state = {'x': [0.0, 200.0], 'y': [0.0, 200.0], 'theta': [0.0, 0.0]}
        self.obs, _, self.done, _ = self.racecar_env.reset(initial_state)
        self.ego_pose = [0., 0., 0.]
        self.ego_speed = [0., 0., 0.]
        self.ego_steer = 0.0
        self.opp_pose = [200., 200., 0.]
        self.opp_speed = [0., 0., 0.]
        self.opp_steer = 0.0

        # keep track of latest sim state
        self.ego_scan = list(self.obs['scans'][0])

        # keep track of collision
        self.ego_collision = False
        self.opp_collision = False

        # transform broadcaster
        self.br = transform_broadcaster.TransformBroadcaster()

        # pubs
        self.ego_scan_pub = rospy.Publisher(self.ego_scan_topic, LaserScan, queue_size=1)
        self.ego_odom_pub = rospy.Publisher(self.ego_odom_topic, Odometry, queue_size=1)
        self.opp_odom_pub = rospy.Publisher(self.opp_odom_topic, Odometry, queue_size=1)
        self.info_pub = rospy.Publisher(self.race_info_topic, RaceInfo, queue_size=1)

        # subs
        self.drive_sub = rospy.Subscriber(self.ego_drive_topic, AckermannDriveStamped,
                                          self.drive_callback, queue_size=1)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(0.004), self.timer_callback)

    def update_sim_state(self):
        self.ego_scan = list(self.obs['scans'][0])

        self.ego_pose[0] = self.obs['poses_x'][0]
        self.ego_pose[1] = self.obs['poses_y'][0]
        self.ego_pose[2] = self.obs['poses_theta'][0]
        self.ego_speed[0] = self.obs['linear_vels_x'][0]
        self.ego_speed[1] = self.obs['linear_vels_y'][0]
        self.ego_speed[2] = self.obs['ang_vels_z'][0]

        self.opp_pose[0] = self.obs['poses_x'][1]
        self.opp_pose[1] = self.obs['poses_y'][1]
        self.opp_pose[2] = self.obs['poses_theta'][1]
        self.opp_speed[0] = self.obs['linear_vels_x'][1]
        self.opp_speed[1] = self.obs['linear_vels_y'][1]
        self.opp_speed[2] = self.obs['ang_vels_z'][1]

    def drive_callback(self, drive_msg):
        # print('in drive callback')
        # TODO: trigger opp agent plan, step env, update pose and steer and vel
        ego_speed = drive_msg.drive.speed
        self.ego_steer = drive_msg.drive.steering_angle
        # opp_speed, self.opp_steer = self.opp_agent.plan(self.obs)
        opp_speed = 0.
        opp_steer = 0.
        action = {
            'ego_idx': 0, 
            'speed': [ego_speed, opp_speed], 
            'steer': [self.ego_steer, self.opp_steer]
        }
        self.obs, step_reward, self.done, info = self.racecar_env.step(action)

        self.update_sim_state()

    def timer_callback(self, timer):
        ts = rospy.Time.now()

        # pub scan
        scan = LaserScan()
        scan.header.stamp = ts
        scan.header.frame_id = 'ego_racecar/laser'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.
        scan.range_max = 30.
        scan.ranges = self.ego_scan
        self.ego_scan_pub.publish(scan)

        # pub tf
        self.publish_odom(ts)
        self.publish_transforms(ts)
        self.publish_laser_transforms(ts)
        self.publish_wheel_transforms(ts)

        # pub race info
        self.publish_race_info(ts)

    def publish_race_info(self, ts):
        info = RaceInfo()
        info.header.stamp = ts
        self.ego_collision = self.ego_collision or self.obs['collisions'][0]
        self.opp_collision = self.opp_collision or self.obs['collisions'][1]
        info.ego_collision = self.ego_collision
        info.opp_collision = self.opp_collision
        info.ego_elapsed_time = self.obs['lap_times'][0]
        info.opp_elapsed_time = self.obs['lap_times'][1]
        info.ego_lap_count = self.obs['lap_counts'][0]
        info.opp_lap_count = self.obs['lap_counts'][1]
        self.info_pub.publish(info)

    def publish_odom(self, ts):
        def create_odom(namespace, stamp, pose, speed):
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = '/map'
            odom.child_frame_id = '%s/base_link' % namespace

            # calculate the position
            odom.pose.pose.position.x = pose[0]
            odom.pose.pose.position.y = pose[1]
            odom.pose.pose.orientation = rotation_from_steer(pose[2])

            odom.twist.twist.linear.x = speed[0]
            odom.twist.twist.linear.y = speed[1]
            odom.twist.twist.linear.z = speed[2]

            return odom

        ego_odom = create_odom("ego_racecar", ts, self.ego_pose, self.ego_speed)
        opp_odom = create_odom("opp_racecar", ts, self.opp_pose, self.opp_speed)

        self.ego_odom_pub.publish(ego_odom)
        self.opp_odom_pub.publish(opp_odom)

    def publish_transforms(self, ts):
        def create_ts(namespace, stamp, pose):
            tl = Vector3()
            tl.x, tl.y, tl.z = pose[0], pose[1], 0.0

            # create the transform
            # Why we're using pose[2] here? Why it's special
            transform = Transform()
            transform.translation = tl
            transform.rotation = rotation_from_steer(pose[2])

            ts = TransformStamped()
            ts.transform = transform
            ts.header.stamp = stamp
            ts.header.frame_id = '/map'
            ts.child_frame_id = '%s/base_link' % namespace

        # I think that create & send are not combined to make sure
        ego_ts = create_ts('ego_racecar', ts, self.ego_pose)
        opp_ts = create_ts('opp_racecar', ts, self.opp_pose)

        self.br.sendTransform(ego_ts)
        self.br.sendTransform(opp_ts)

    def publish_wheel_transforms(self, ts):
        def send_wheel_ts(namespace, stamp, steer):

            # calculate wheel rotation from steer
            wheel_ts = TransformStamped()
            wheel_ts.transform.rotation = rotation_from_steer(steer)
            wheel_ts.header.stamp = stamp

            # left & right side of the front wheel has the same angle
            # (hence same transform)
            for side in ['left', 'right']:
                wheel_ts.header.frame_id = "%s/front_%s_hinge" % (namespace, side)
                wheel_ts.header.frame_id = "%s/front_%s_hinge" % (namespace, side)
                self.br.sendTransform(wheel_ts)

        send_wheel_ts('ego_racecar', ts, self.ego_steer)
        send_wheel_ts('opp_racecar', ts, self.opp_steer)

    def publish_laser_transforms(self, ts):
        def send_scan_ts(namespace, stamp, distance_to_base):
            """
            Creates a TransformStamped (Timestamped Geometry Transform) object
            for the laser scan on Cars

            This function actually does nothing, since geometry of laser relative
            to base does not change. So I'm not sure why we have it.
            """
            # TODO: check frame names
            scan_ts = TransformStamped()
            scan_ts.transform.translation.x = distance_to_base
            scan_ts.transform.rotation.w = 1.
            scan_ts.header.stamp = stamp
            scan_ts.header.frame_id = '%s/base_link' % namespace
            scan_ts.child_frame_id = '%s/laser' % namespace
            self.br.sendTransform(scan_ts)

        send_scan_ts("ego_racecar", ts, self.scan_distance_to_base_link)
        send_scan_ts("opp_racecar", ts, self.scan_distance_to_base_link)


if __name__ == '__main__':
    rospy.init_node('gym_bridge')
    gym_bridge = GymBridge()
    rospy.spin()
