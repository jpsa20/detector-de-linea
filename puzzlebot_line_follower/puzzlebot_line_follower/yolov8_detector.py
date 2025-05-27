#!/usr/bin/env python3
# File: puzzlebot_line_follower/yolov8_detector.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolov8_msgs.msg import InferenceResult, Yolov8Inference
from ultralytics import YOLO
import cv2

class Yolov8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # Parámetros (ajústalos en tu YAML)
        self.declare_parameter('model_path',
                               '/home/jp/ros2_ws_2/src/puzzlebot_line_follower/models/lastfinal.pt')
        self.declare_parameter('image_topic', '/video_source/raw')
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('inference_topic', '/yolov8/inference')
        self.declare_parameter('annotated_image_topic', '/inference_result')

        model_path  = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        conf_thresh = self.get_parameter('confidence_threshold').value
        inf_topic   = self.get_parameter('inference_topic').value
        img_topic   = self.get_parameter('annotated_image_topic').value

        # Carga de modelo y bridge
        self.model  = YOLO(model_path)
        self.bridge = CvBridge()
        self.conf_thresh = conf_thresh

        # Publicadores
        self.inf_pub = self.create_publisher(Yolov8Inference, inf_topic, 1)
        self.img_pub = self.create_publisher(Image, img_topic, 1)

        # Suscripción a la cámara
        self.create_subscription(Image, image_topic,
                                 self.camera_callback, 10)

        self.get_logger().info(
            f'[Yolov8Detector] Cargando modelo: {model_path} · '
            f'Suscrito a: {image_topic}'
        )

    def camera_callback(self, msg: Image):
        # 1) ROS Image → OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 2) Inferencia YOLOv8
        results = self.model(frame)[0]

        # 3) Montar mensaje Yolov8Inference
        inf_msg = Yolov8Inference()
        inf_msg.header.stamp    = self.get_clock().now().to_msg()
        inf_msg.header.frame_id = msg.header.frame_id

        # 4) Procesar detecciones
        for box in results.boxes:
            conf = float(box.conf.cpu().numpy())
            if conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
            cls = int(box.cls.cpu().numpy())

            res = InferenceResult()
            res.class_name = self.model.names[cls]
            res.top        = y1
            res.left       = x1
            res.bottom     = y2
            res.right      = x2
            inf_msg.yolov8_inference.append(res)

            # Dibujar caja y etiqueta
            color = (0,255,0) if res.class_name=='green' else \
                    (0,0,255) if res.class_name=='red'   else \
                    (0,255,255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{res.class_name} {int(conf*100)}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        # 5) Publicar resultados
        out_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_img.header = inf_msg.header

        self.img_pub.publish(out_img)
        self.inf_pub.publish(inf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
