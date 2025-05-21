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

        # Suscripción dinámica al tópico de cámara
        self.get_logger().info(f'Suscribiendo a imágenes en: {self.image_topic}')
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.cb_image, 1)

        # Publishers:
        # 1) Ahora publicamos la imagen binarizada en lugar de la ROI cruda
        self.bin_image_pub   = self.create_publisher(Image,   '/line_detector/roi_image',   1)
        # 2) Imagen de debug con líneas detectadas
        self.debug_image_pub = self.create_publisher(Image,   '/line_detector/debug_image', 1)
        # 3) Error lateral en píxeles
        self.error_pub       = self.create_publisher(Float32, '/line_detector/error',       1)

        self.get_logger().info('LineLightDetector iniciado y esperando imágenes...')

    def cb_image(self, msg: Image):
        # 1) Convertir ROS Image a OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 2) Redimensionar a tamaño fijo
        target_width, target_height = 720, 480
        frame = cv2.resize(frame, (target_width, target_height))

        # 3) Recortar ROI inferior
        h, w = frame.shape[:2]
        y0 = int(h * self.roi_y_start)
        roi = frame[y0:, :]
        rh, rw = roi.shape[:2]

        # 4) Crear máscara trapezoidal
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

        # 5) Preprocesado: gris → CLAHE → blur
        gray       = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)
        gray_clahe = self.clahe.apply(gray)
        blurred    = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

        # 6) Umbral inverso Otsu
        _, binary_inv = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 7) Publicar aquí la imagen **binarizada**
        bin_bgr = cv2.cvtColor(binary_inv, cv2.COLOR_GRAY2BGR)
        bin_msg = self.bridge.cv2_to_imgmsg(bin_bgr, encoding='bgr8')
        bin_msg.header = msg.header
        self.bin_image_pub.publish(bin_msg)

        # 8) Morfología
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        morph = cv2.erode(binary_inv, kern, iterations=self.morph_iter)
        morph = cv2.dilate(morph, kern, iterations=self.morph_iter)

        # 9) Detección de bordes y recortes
        edges = cv2.Canny(morph, 50, 150)
        crop = int(rw * 0.05)
        side_mask = np.zeros_like(edges)
        cv2.rectangle(side_mask, (crop, 0), (rw - crop, rh), 255, -1)
        edges = cv2.bitwise_and(edges, edges, mask=side_mask)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        # 10) Transformada Hough probabilística
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=50, minLineLength=5, maxLineGap=50)

        # 11) Dibujar líneas y calcular error
        debug = roi.copy()
        error = 0.0
        if lines is not None and len(lines) > 0:
            sum_x, count = 0.0, 0
            for l in lines:
                x1, y1, x2, y2 = l[0]
                cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
                sum_x += (x1 + x2) / 2
                count += 1
            avg_x = sum_x / count
            error = float(avg_x - rw / 2)
            self.get_logger().info(f'Viendo línea: error = {error:.1f} px')

        # 12) Publicar debug_image y error
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
