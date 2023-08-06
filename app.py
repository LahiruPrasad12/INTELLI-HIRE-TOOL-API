from flask import Flask, request, redirect, jsonify
from static.logger import logging
from controllers.auth_controller import auth_controller
from controllers.ethical_benchmark_controller import ethical_benchmark_controller

app = Flask(__name__)

logging.info(f'Preprocessed Text : {"Flask Server is started"}')

app.register_blueprint(auth_controller, url_prefix='/auth')
app.register_blueprint(ethical_benchmark_controller, url_prefix='/ethical_benchmark')


if __name__ == "__main__":
    app.run()