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
        self.bridge = CvBridge()

        # Parámetros de visión
        self.declare_parameter('vision.image_topic', '/video_source/raw')
        self.declare_parameter('vision.roi_y_start', 0.4)
        self.declare_parameter('vision.trap_top_ratio', 0.9)
        self.declare_parameter('vision.morph_iter', 2)

        vp = self.get_parameter
        self.image_topic    = vp('vision.image_topic').value
        self.roi_y_start    = vp('vision.roi_y_start').value
        self.trap_top_ratio = vp('vision.trap_top_ratio').value
        self.morph_iter     = vp('vision.morph_iter').value

        # CLAHE para contraste local
        self.clahe = cv2.createCLAHE(clipLimit=2.45, tileGridSize=(7, 7))

        # Suscripción al tópico de imagen
        self.get_logger().info(f'Suscribiendo a imágenes en: {self.image_topic}')
        self.create_subscription(
            Image, self.image_topic, self.cb_image, 1)

        # Publishers
        self.roi_image_pub   = self.create_publisher(Image,   '/line_detector/roi_image',   1)
        self.debug_image_pub = self.create_publisher(Image,   '/line_detector/debug_image', 1)
        self.error_pub       = self.create_publisher(Float32, '/line_detector/error',       1)

        self.get_logger().info('LineLightDetector iniciado y esperando imágenes...')

    def cb_image(self, msg: Image):
        # Convertir ROS Image a OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Redimensionar para consistencia
        target_width, target_height = 720, 480
        frame = cv2.resize(frame, (target_width, target_height))

        # Recortar ROI inferior
        h, w = frame.shape[:2]
        y0 = int(h * self.roi_y_start)
        roi = frame[y0:, :]
        rh, rw = roi.shape[:2]

        # Publicar ROI cruda (opcional)
        roi_msg = self.bridge.cv2_to_imgmsg(roi, encoding='bgr8')
        roi_msg.header = msg.header
        self.roi_image_pub.publish(roi_msg)

        # Crear máscara trapezoidal
        top_w = int(rw * self.trap_top_ratio)
        trap = np.array([[
            ((rw - top_w) // 2, 0),
            ((rw + top_w) // 2, 0),
            (rw, rh),
            (0, rh)
        ]], dtype=np.int32)
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(mask, trap, 255)
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

        # Gris → CLAHE → blur
        gray       = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)
        gray_clahe = self.clahe.apply(gray)
        blurred    = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

        # Umbral inverso Otsu
        _, binary_inv = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morfología
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        morph = cv2.erode(binary_inv, kern, iterations=self.morph_iter)
        morph = cv2.dilate(morph, kern, iterations=self.morph_iter)

        # Canny + recortes lateral y trapezoidal
        edges = cv2.Canny(morph, 50, 150)
        crop = int(rw * 0.05)
        side_mask = np.zeros_like(edges)
        cv2.rectangle(side_mask, (crop, 0), (rw - crop, rh), 255, -1)
        edges = cv2.bitwise_and(edges, edges, mask=side_mask)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        # HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=5,
            maxLineGap=50
        )

        debug = roi.copy()
        error = 0.0

        if lines is not None:
            left_x_vals, right_x_vals, center_x_vals = [], [], []
            for l in lines:
                x1, y1, x2, y2 = l[0]
                # Dibujar y clasificar por pendiente
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                mid_x = (x1 + x2) / 2.0
                if abs(slope) < 0.3:
                    center_x_vals.append(mid_x)
                    color = (255, 255, 0)  # amarillo para central
                elif slope < 0:
                    left_x_vals.append(mid_x)
                    color = (0, 0, 255)    # rojo para izquierda
                else:
                    right_x_vals.append(mid_x)
                    color = (0, 255, 0)    # verde para derecha
                cv2.line(debug, (x1, y1), (x2, y2), color, 2)

            # Fallback central → laterales
            if center_x_vals:
                target_x = sum(center_x_vals) / len(center_x_vals)
            elif left_x_vals and right_x_vals:
                left_x  = sum(left_x_vals)  / len(left_x_vals)
                right_x = sum(right_x_vals) / len(right_x_vals)
                target_x = (left_x + right_x) / 2.0
            else:
                target_x = rw / 2.0  # fallback al centro

            error = float(target_x - rw/2.0)
            cv2.circle(debug, (int(target_x), rh//2), 5, (0, 255, 255), -1)
            self.get_logger().info(f'Error carril: {error:.1f} px')

        # Publicar debug y error
        dbg_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        dbg_msg.header = msg.header
        self.debug_image_pub.publish(dbg_msg)
        self.error_pub.publish(Float32(data=error))

def main(args=None):
    rclpy.init(args=args)
    node = LineLightDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
