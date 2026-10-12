import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class LidarConverter(Node):

    def __init__(self):
        super().__init__('lidar_converter')
        self.publisher_ = self.create_publisher(PointCloud2, '/points_raw_time', 10)
        # self.publisher_ = self.create_publisher(PointCloud2, '/summit_xl_1/top_lidar/points_time', 10)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/points_raw',
            self.timer_callback,
            10)
        # self.subscription = self.create_subscription(
        #     PointCloud2,
        #     '/summit_xl_1/top_lidar/points',
        #     self.timer_callback,
        #     10)

    def timer_callback(self,msg):

        # available list of fields , espcially if there is not ring on the lsit we will calculate it 
        available_fields = [f.name for f in msg.fields]
        field_names = ["x", "y", "z", "intensity"]
        if "ring" in available_fields:
            field_names.append("ring")

        # reading 
        points = pc2.read_points(msg, field_names=tuple(field_names), skip_nans=True)
        # points = pc2.read_points(msg, field_names=("x", "y", "z", "intensity","ring"), skip_nans=True)
        points = np.array(list(points))
        # print(points)
        x,y,z,intensity=[],[],[],[]
        curr_ring= []
        # print(len(points[0]))
        if len(points[0]) == 5 : # ring available 
            for i in points :
                # print(i[0])
                    x.append(i[0])
                    y.append(i[1])
                    z.append(i[2])
                    intensity.append(i[3])
                    curr_ring.append(i[4])                   
        else:   #  no ring    
            for i in points :
                    # print(i[0])
                    x.append(i[0])
                    y.append(i[1])
                    z.append(i[2])
                    intensity.append(i[3])
        

        # velodyne 64 scan, 10 Hz, 0.1s , si j'ai bien compris on a besoin d'ajouter ring et time. La "date" pour chaque points acquis et ring c'est le lequel des 64 couches de scan est à l'origine de ce point

        # time
        horizontal_angle= np.arctan2(y,x)
        # 2pi --> 0.1s, +pi because atan2 sendback an angle between -pi and +pi
        time = np.nan_to_num((horizontal_angle + np.pi) / (2 * np.pi) * 0.1, nan=0.0) # pour enlevr nan 

        # ring 
        #-24.8 et +2 degree represented by 64 scan 0 to 63, So -24.8 deg corresponds to the ring 0 and +2 deg corresponds to the ring 63
        max=2
        min=-24.8
        max_ring=63 
        range=max-min
        vertical_angle= np.arctan2(z,(np.sqrt(np.multiply(x,x) + np.multiply(y,y))))
        vertical_angle_degree= vertical_angle*180/np.pi
        # print(f"{vertical_angle_degree}")
        ring= (vertical_angle_degree -min) *max_ring / (range) # - min (+24.8) pour initialiser a 0, diviser par range car 2 + initilisation 24.8 : range total
        ring = ring.astype(np.uint16) # convert to int
        ring = np.clip(ring,0,63) # for security if there is more than 63 or under 0

        new_points = np.zeros(len(x), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('ring', np.uint16),
            ('time', np.float32),
        ])

        new_points['x'] = x
        new_points['y'] = y
        new_points['z'] = z
        new_points['intensity'] = intensity
        if len(points[0])==5:
             new_points['ring'] = curr_ring
            #  print('on garde l ancien ring')
        else:
             new_points['ring'] = ring
             
        new_points['time'] = time

        # print(new_points)
        msg_pub = pc2.create_cloud(msg.header, self.get_fields(), new_points)
        msg_pub.header.frame_id = "summit_xl_1/summit_xl_top_lidar_link"
        msg_pub.is_dense = True
        self.publisher_.publish(msg_pub)


    def get_fields(self):
        return [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring',      offset=16, datatype=PointField.UINT16,  count=1),
            # offset 18 + 2 bytes de padding → alignement 4 bytes
            PointField(name='time',      offset=18, datatype=PointField.FLOAT32, count=1),
        ]

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = LidarConverter()

    rclpy.spin(minimal_publisher)


    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
