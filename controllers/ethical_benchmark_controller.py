from flask import Blueprint, jsonify, Response
from pipe.ethical_benchmark_pipeline import EthicalBenchMarkDetector
import cv2

ethical_benchmark_controller = Blueprint('ethical_benchmark', __name__)

@ethical_benchmark_controller.route("/")
def video_feed():
    return Response(EthicalBenchMarkDetector().run(), mimetype='multipart/x-mixed-replace; boundary=frame')
