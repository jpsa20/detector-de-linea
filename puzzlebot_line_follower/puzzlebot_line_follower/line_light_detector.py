#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

class LineLightDetector(Node):
    def __init__(self):
        super().__init__('line_light_detector')
        # parámetros
        self.declare_parameter('image_topic',      '/video_source/raw')
        self.declare_parameter('roi_y_start',       0.4)
        self.declare_parameter('x_crop_ratio',      0.2)
        self.declare_parameter('morph_kernel_size', 3)
        self.declare_parameter('morph_iterations',  2)
        self.declare_parameter('use_clahe',        True)
        self.declare_parameter('bottom_tolerance',  2)

        # publishers & bridge
        self.pub_error = self.create_publisher(Float32, '/line_detector/error', 10)
        self.pub_debug = self.create_publisher(Image,    '/line_detector/debug_image', 10)
        self.bridge    = CvBridge()

        # suscripción al tópico de imagen
        image_topic = self.get_parameter('image_topic').value
        self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.get_logger().info(f'Subscrito a {image_topic}')

    def image_callback(self, msg: Image):
        # 1) Convertir ROS Image → BGR y rotar 180°
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        Hf, Wf = frame.shape[:2]

        # 2) Recortar ROI rectangular (solo zona donde está la pista)
        y0     = int(self.get_parameter('roi_y_start').value * Hf)
        x_crop = int(self.get_parameter('x_crop_ratio').value * Wf)
        roi    = frame[y0:Hf, x_crop:Wf - x_crop]
        h, w   = roi.shape[:2]

        # 3) Gris + blur + CLAHE
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        if self.get_parameter('use_clahe').value:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray  = clahe.apply(gray)

        # 4) Binarización inversa Otsu + morfología de cierre
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        ksize = self.get_parameter('morph_kernel_size').value
        kernel = np.ones((ksize, ksize), np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel,
            iterations=self.get_parameter('morph_iterations').value
        )

        # 5) Encontrar todos los contornos
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 6) Filtrar contornos que “tocan” el fondo de la ROI
        tol = self.get_parameter('bottom_tolerance').value
        valid = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if y + ch >= h - tol:
                valid.append(cnt)

        # 7) Si no hay contornos válidos, publica error cero
        if not valid:
            self.get_logger().warn('No se detectó línea; publicando error=0')
            self.pub_error.publish(Float32(data=0.0))
            return

        # 8) Calcular centroides de contornos válidos
        centroids = []
        for cnt in valid:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            centroids.append((cnt, cx))

        if not centroids:
            self.get_logger().warn('Contornos sin área; publicando error=0')
            self.pub_error.publish(Float32(data=0.0))
            return

        # 9) Seleccionar los dos centroides más cercanos al centro de la ROI
        x_center = w / 2.0
        centroids.sort(key=lambda c: abs(c[1] - x_center))
        selected = centroids[:2]

        # 10) Promediar sus coordenadas x para obtener el eje de la línea
        avg_cx = sum(c[1] for c in selected) / len(selected)

        # 11) Calcular error lateral normalizado y publicarlo
        error_px   = x_center - avg_cx
        error_norm = float(error_px) / x_center
        self.pub_error.publish(Float32(data=error_norm))

        # Mensajes de depuración de dirección
        if error_norm > 0:
            self.get_logger().info(f'Acomodando hacia la IZQUIERDA (error={error_norm:.3f})')
        elif error_norm < 0:
            self.get_logger().info(f'Acomodando hacia la DERECHA  (error={error_norm:.3f})')
        else:
            self.get_logger().info('Centrado en la línea (error≈0)')

        # 12) Imagen de depuración
        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cnt, cx in selected:
            cv2.drawContours(debug, [cnt], -1, (0,255,0), 2)
            # centroid y coordenada y aproximada
            y_vals = [pt[0][1] for pt in cnt]
            cy = int(sum(y_vals) / len(y_vals))
            cv2.circle(debug, (int(cx), cy), 5, (0,0,255), -1)
        # línea promedio y centro de imagen
        cv2.line(debug, (int(avg_cx), 0), (int(avg_cx), h), (255,0,255), 2)
        cv2.line(debug, (int(x_center), 0), (int(x_center), h), (255,0,0),    2)

        debug_msg = self.bridge.cv2_to_imgmsg(debug, 'bgr8')
        self.pub_debug.publish(debug_msg)

    def destroy_node(self):
        self.get_logger().info('Cerrando nodo')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LineLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
