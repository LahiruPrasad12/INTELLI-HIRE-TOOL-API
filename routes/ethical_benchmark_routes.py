from flask import Blueprint, jsonify

auth_route = Blueprint('auth', __name__)

@auth_route.route("/login")
def login():
    return jsonify(message="Login route")

@auth_route.route("/register")
def register():
    return jsonify(message="Register route")