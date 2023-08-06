from flask import Blueprint, jsonify

auth_controller = Blueprint('auth', __name__)

@auth_controller.route("/login")
def login():
    return jsonify(message="Login route")

@auth_controller.route("/register")
def register():
    return jsonify(message="Register route")