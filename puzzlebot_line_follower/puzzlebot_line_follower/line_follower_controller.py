#!/usr/bin/env python3
# File: puzzlebot_line_follower/line_follower_controller.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time

class LineFollowerController(Node):
    def __init__(self):
        super().__init__('line_follower_controller')

        # Parámetros dinámicos (ajusta en tu YAML)
        self.declare_parameter('error_topic', '/line_detector/error')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('loop_hz', 10.0)
        self.declare_parameter('Kp', 1.0)
        self.declare_parameter('Ki', 0.0)
        self.declare_parameter('Kd', 0.1)
        self.declare_parameter('v_default', 0.15)            # velocidad lineal base [m/s]
        self.declare_parameter('v_caution_factor', 0.5)     # factor reducción en amarillo
        self.declare_parameter('max_angular_vel', 1.0)      # límite ω [rad/s]

        # Obtener parámetros
        self.error_topic     = self.get_parameter('error_topic').value
        self.cmd_vel_topic   = self.get_parameter('cmd_vel_topic').value
        self.loop_hz         = self.get_parameter('loop_hz').value
        self.Kp              = self.get_parameter('Kp').value
        self.Ki              = self.get_parameter('Ki').value
        self.Kd              = self.get_parameter('Kd').value
        self.v_default       = self.get_parameter('v_default').value
        self.v_caution       = self.v_default * self.get_parameter('v_caution_factor').value
        self.max_omega       = self.get_parameter('max_angular_vel').value

        # Estado de semáforo (por ahora siempre GO)
        self.state = 'GO'  # podrá cambiarse cuando añadas Yolov8

        # Variables PID
        self.error       = 0.0
        self.prev_error  = 0.0
        self.integral    = 0.0
        self.prev_time   = None

        # Publicador de cmd_vel
        self.pub_vel = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Suscriptor al error de línea
        self.create_subscription(Float32, self.error_topic, self.error_callback, 10)

        # Timer de control
        period = 1.0 / self.loop_hz
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(f'[{self.get_name()}] Inicializado: suscrito a {self.error_topic}, publicando en {self.cmd_vel_topic} a {self.loop_hz} Hz')

    def error_callback(self, msg: Float32):
        self.error = msg.data

    def control_loop(self):
        # Tiempo actual en nanosegundos
        now = self.get_clock().now()
        now_ns = now.nanoseconds

        # Primer ciclo: asumimos periodo constante
        if self.prev_time is None:
            dt = 1.0 / self.loop_hz
        else:
            # Calculamos dt en segundos a partir de la diferencia de nanosegundos
            dt = (now_ns - self.prev_time) * 1e-9

        # Guardamos timestamp para el siguiente ciclo
        self.prev_time = now_ns

        # PID
        self.integral += self.error * dt
        derivative = (self.error - self.prev_error) / dt if dt > 0 else 0.0
        omega = (self.Kp * self.error +
                 self.Ki * self.integral +
                 self.Kd * derivative)
        self.prev_error = self.error

        # Limitamos omega
        omega = max(min(omega, self.max_omega), -self.max_omega)

        # Selección de velocidad lineal según estado
        if self.state == 'GO':
            v = self.v_default
        elif self.state == 'CAUTION':
            v = self.v_caution
        else:  # 'STOP'
            v = 0.0
            omega = 0.0

        # Publicar cmd_vel
        twist = Twist()
        twist.linear.x  = v
        twist.angular.z = omega
        self.pub_vel.publish(twist)


    def destroy_node(self):
        self.get_logger().info(f'[{self.get_name()}] Cerrando nodo controlador')
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
