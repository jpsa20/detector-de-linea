#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class LineFollowerController(Node):
    def __init__(self):
        super().__init__('line_follower_controller')

        # --- Parámetros de velocidad y PID ---
        self.declare_parameter('control.speed.default', 0.15)
        self.declare_parameter('control.pid.Kp', 0.5)
        self.declare_parameter('control.pid.Ki', 0.0)
        self.declare_parameter('control.pid.Kd', 0.1)

        self.declare_parameter('control.max_v', 0.3)
        self.declare_parameter('control.min_v', 0.0)
        self.declare_parameter('control.max_w', 0.5)
        self.declare_parameter('control.min_w', 0.05)

        p = self.get_parameter
        self.v_default = p('control.speed.default').value
        self.Kp = p('control.pid.Kp').value
        self.Ki = p('control.pid.Ki').value
        self.Kd = p('control.pid.Kd').value
        self.max_v = p('control.max_v').value
        self.min_v = p('control.min_v').value
        self.max_w = p('control.max_w').value
        self.min_w = p('control.min_w').value

        # Variables internas PID
        self.error = 0.0
        self.prev_error = 0.0
        self.integral = 0.0

        self.dt = 0.1  # 10 Hz

        # Suscripción solo al error lateral
        self.create_subscription(
            Float32, '/line_detector/error', self.error_cb, 1)

        # Publisher para cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # Opcional: publicar error y omega para debug
        self.debug_err_pub = self.create_publisher(Float32, '/line_follower/error', 1)
        self.debug_omega_pub = self.create_publisher(Float32, '/line_follower/omega', 1)

        # Timer de control
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('LineFollowerController iniciado.')

    def error_cb(self, msg: Float32):
        self.error = msg.data

    def control_loop(self):
        # Calcular PID angular
        de = (self.error - self.prev_error) / self.dt
        self.integral += self.error * self.dt
        omega = self.Kp * self.error + self.Ki * self.integral + self.Kd * de
        self.prev_error = self.error

        # Debug
        self.debug_err_pub.publish(Float32(data=self.error))
        self.debug_omega_pub.publish(Float32(data=omega))

        # Velocidad lineal constante
        v = self.v_default

        # Zonas muertas y saturación
        if abs(omega) < self.min_w:
            omega = 0.0
        v = max(self.min_v, min(v, self.max_v))
        omega = max(-self.max_w, min(omega, self.max_w))

        # Publicar comando de velocidad
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = omega
        self.cmd_pub.publish(twist)

    def destroy_node(self):
        self.get_logger().info('LineFollowerController detenido.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
