from flask import Blueprint, jsonify
from pipe.ethical_benchmark_pipeline import EthicalBenchMarkDetector
import cv2

ethical_benchmark_controller = Blueprint('ethical_benchmark', __name__)

@ethical_benchmark_controller.route("/")
def loadAPP():
    cap = cv2.VideoCapture(0)
    detector = EthicalBenchMarkDetector()
    detector.run(cap)
    cap.release()
    return "success"
