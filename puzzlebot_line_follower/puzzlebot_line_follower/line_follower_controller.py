#!/usr/bin/env python3
# File: puzzlebot_line_follower/line_follower_controller.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class LineFollowerController(Node):
    def __init__(self):
        super().__init__('line_follower_controller')

        # 1) Declaración de parámetros
        self.declare_parameter('control.error_topic',    '/line_detector/error')
        self.declare_parameter('control.cmd_vel_topic',  '/cmd_vel')
        self.declare_parameter('control.loop_hz',         30.0)
        self.declare_parameter('control.speed.default',   0.10)
        self.declare_parameter('control.pid.Kp',          0.05)
        self.declare_parameter('control.pid.Ki',          0.0)
        self.declare_parameter('control.pid.Kd',          0.01)
        self.declare_parameter('control.max_v',           0.15)
        self.declare_parameter('control.min_v',           0.0)
        self.declare_parameter('control.max_w',            1.0)
        self.declare_parameter('control.error_deadzone',   0.0)

        # 2) Lectura de parámetros
        p = self.get_parameter
        self.error_topic    = p('control.error_topic').value
        self.cmd_vel_topic  = p('control.cmd_vel_topic').value
        self.loop_hz        = p('control.loop_hz').value
        self.v_default      = p('control.speed.default').value
        self.Kp             = p('control.pid.Kp').value
        self.Ki             = p('control.pid.Ki').value
        self.Kd             = p('control.pid.Kd').value
        self.max_v          = p('control.max_v').value
        self.min_v          = p('control.min_v').value
        self.max_w          = p('control.max_w').value
        self.dz_frac        = p('control.error_deadzone').value / 100.0

        # 3) Estado interno del PID
        self.error       = 0.0
        self.prev_error  = 0.0
        self.integral    = 0.0

        # Historial de errores
        self.current_error    = 0.0
        self.prev_error_1     = 0.0
        self.prev_error_2     = 0.0
        self.received_error   = False

        # 4) Suscripción al tópico de error
        self.create_subscription(Float32, self.error_topic, self.error_cb, 10)

        # 5) Publicador de cmd_vel
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # 6) Timer de control
        self.dt = 1.0 / self.loop_hz
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f'[{self.get_name()}] Controlador listo: '
            f'subscrito a "{self.error_topic}", publicando en "{self.cmd_vel_topic}" a {self.loop_hz} Hz'
        )

    def error_cb(self, msg: Float32):
        # Desplazamiento del historial
        self.prev_error_2 = self.prev_error_1
        self.prev_error_1 = self.current_error
        self.current_error = msg.data
        self.received_error = True

    def control_loop(self):
        # Usa el error actual si se recibió, si no, retrocede dos pasos
        if self.received_error:
            e = self.current_error
            self.received_error = False
        else:
            e = self.prev_error_2
            self.get_logger().warn("No se recibió error nuevo. Usando error retrasado.")

        # 1) Aplica zona muerta
        if abs(e) < self.dz_frac:
            e = 0.0

        # 2) PID
        de       = (e - self.prev_error) / self.dt
        self.integral += e * self.dt
        omega    = (self.Kp * e +
                    self.Ki * self.integral +
                    self.Kd * de)
        self.prev_error = e

        # 3) Saturación angular
        omega = max(-self.max_w, min(omega, self.max_w))

        # 4) Escala velocidad lineal según |omega|
        scale = 1.0 - min(abs(omega) / self.max_w, 1.0)
        v     = self.v_default * scale
        v     = max(self.min_v, min(v, self.max_v))

        # 5) Publica cmd_vel
        twist = Twist()
        twist.linear.x  = v
        twist.angular.z = omega
        self.cmd_pub.publish(twist)

    def destroy_node(self):
        self.get_logger().info(f'[{self.get_name()}] Controlador detenido.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()