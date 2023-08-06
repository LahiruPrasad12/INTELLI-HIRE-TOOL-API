from flask import Flask, request, redirect, jsonify
from static.logger import logging
from routes.auth_route import auth_route


app = Flask(__name__)

logging.info(f'Preprocessed Text : {"Flask Server is started"}')

app.register_blueprint(auth_route, url_prefix='/auth')


if __name__ == "__main__":
    app.run()