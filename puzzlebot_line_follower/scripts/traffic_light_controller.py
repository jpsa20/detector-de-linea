#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetLightProperties

class TrafficLightController(Node):
    def __init__(self):
        super().__init__('traffic_light_controller')
        self.cli = self.create_client(SetLightProperties, '/gazebo/set_light_properties')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Esperando servicio /gazebo/set_light_properties…')
        self.states = [
            ('traffic_light::red_light',    (1,0,0,1), 5.0),
            ('traffic_light::yellow_light', (1,1,0,1), 2.0),
            ('traffic_light::green_light',  (0,1,0,1), 5.0),
        ]
        self.idx = 0
        self.timer = self.create_timer(0.1, self.tick)

    def tick(self):
        name, color, duration = self.states[self.idx]
        req = SetLightProperties.Request()
        req.light_name = name
        req.diffuse.r, req.diffuse.g, req.diffuse.b, req.diffuse.a = color
        req.attenuation_constant = 0.5
        self.cli.call_async(req)
        self.get_logger().info(f'{name} → {color}')
        # planear siguiente estado
        self.timer.cancel()
        self.create_timer(duration, self.next_state)

    def next_state(self):
        self.idx = (self.idx + 1) % len(self.states)
        self.timer = self.create_timer(0.1, self.tick)

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
