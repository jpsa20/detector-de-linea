#!/usr/bin/env python3
import cv2
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolov8_msgs.msg import InferenceResult, Yolov8Inference

class Yolo8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # Parámetro para la ruta del modelo (puede remapearse en YAML)
        default_path = '/home/jp/ros2_ws_2/src/puzzlebot_line_follower/models/lastfinal.pt'
        self.declare_parameter('model_path', default_path)
        model_path = self.get_parameter('model_path').value

        # Carga el modelo y el bridge
        self.model  = YOLO(model_path)
        self.bridge = CvBridge()

        # Colores para cada clase detectada
        self.class_colors = {
            0: (0, 255, 0),    # e.g. green light
            1: (0, 0, 255),    # e.g. red light
            3: (0, 255, 255),  # e.g. yellow light
        }

        # Subscribirse a la cámara
        self.create_subscription(
            Image, '/image_raw', self.camera_callback, 10)

        # Publicadores: imagen anotada y resultados estructurados
        self.img_pub = self.create_publisher(Image,           '/inference_result', 1)
        self.inf_pub = self.create_publisher(Yolov8Inference, '/yolov8/inference', 1)

        self.get_logger().info(f"Yolo8Detector cargó el modelo: {model_path}")

    def camera_callback(self, msg: Image):
        # 1) ROS → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 2) Inferencia (solo un frame)
        results = self.model(frame)[0]

        # 3) Prepara mensaje de output
        inf_msg = Yolov8Inference()
        inf_msg.header.stamp    = self.get_clock().now().to_msg()
        inf_msg.header.frame_id = msg.header.frame_id

        # 4) Recorre todas las cajas
        for box in results.boxes:
            conf = float(box.conf.cpu().numpy())
            if conf < 0.6:
                continue

            x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
            cls = int(box.cls.cpu().numpy())

            r = InferenceResult()
            r.class_name = self.model.names[cls]
            r.top        = y1
            r.left       = x1
            r.bottom     = y2
            r.right      = x2
            inf_msg.yolov8_inference.append(r)

            color = self.class_colors.get(cls, (255,255,255))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{r.class_name} {int(conf*100)}%"
            cv2.putText(frame, label,
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        # 5) Publicar imagen anotada
        out_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_img.header = inf_msg.header
        self.img_pub.publish(out_img)

        # 6) Publicar resultados estructurados
        self.inf_pub.publish(inf_msg)

def main():
    rclpy.init()
    node = Yolo8Detector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
