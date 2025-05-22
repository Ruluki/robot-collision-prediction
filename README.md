# Robot Collision Prediction Project

## Project Overview

This project simulates a robot navigating a 2D environment and uses a neural network to predict whether a chosen action will result in a collision with walls.

## Features

- Data loading, preprocessing, and dataset balancing
- Feed-forward neural network for collision prediction based on sensor readings and actions
- 2D physics simulation environment using Pygame and Pymunk
- Robot steering behaviors like Wander and Seek
- Evaluation script to test model predictions in simulation and score accuracy

## Project Structure

- `Data_Loader.py` - Loads and preprocesses training data, balances classes, normalizes features.
- `Networks.py` - Defines and trains the neural network, saves the trained model.
- `grader.py` - Loads the trained model, runs simulation, and evaluates prediction accuracy.
- `SimulationEnvironment.py` - Implements the 2D simulation world and robot sensors.
- `SteeringBehaviors.py` - Contains movement strategies for the robot.
- `Helper.py` - Utility math functions.
- `saved/` - Contains training data (`training_data.csv`), saved model, and scaler.
- `assets/` - (Optional) Contains simulation resources like images or environment assets.
- `requirements.txt` - Python package dependencies.
- `.gitignore` - Files and folders to ignore in version control.
- `LICENSE` - Project license (MIT).

## Requirements

torch==2.2.2+cpu
numpy==1.26.1
scikit-learn==1.5.2
pygame==2.6.1
pymunk==5.7.0
matplotlib==3.7.1
perlin-noise==1.13

Install dependencies with:

```bash
pip install -r requirements.txt
