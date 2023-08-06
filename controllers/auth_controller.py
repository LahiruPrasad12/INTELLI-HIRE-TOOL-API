from flask import Flask, request, redirect, jsonify
from static.logger import logging

def login():
    return jsonify(message="Login route")