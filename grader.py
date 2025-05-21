from SteeringBehaviors import Wander
import SimulationEnvironment as sim
import torch.nn as nn
from Networks import Action_Conditioned_FF

import pickle
import numpy as np
import torch


def get_network_param(sim_env, action, scaler):
    sensor_readings = sim_env.raycasting()
    network_param = np.append(sensor_readings, [action, 0]) #unutilized 0 added to match shape of scaler
    network_param = scaler.transform(network_param.reshape(1,-1))
    network_param = network_param.flatten()[:-1]
    network_param = torch.as_tensor(network_param, dtype=torch.float32)
    return network_param


def evaluate_accuracy(total_actions):
    sim_env = sim.SimulationEnvironment()
    action_repeat = 100
    steering_behavior = Wander(action_repeat)
    # steering_behavior = Seek(sim_env.goal_body.position)

    #load model
    try:
        model = Action_Conditioned_FF()
        model.load_state_dict(torch.load('saved\saved_model.pkl', map_location=torch.device('cpu')))
        model.eval()
        assert isinstance(model, nn.Module), 'Action_Conditioned_FF does not inherit from nn.Module'
    except AssertionError as error:
        print('{"fractionalScore\": %f, \"feedback\": \"%s\"}' % (0.0, error))
        return
    
    except:
        error = 'Error initializing or loading saved_model.pkl'
        print('{"fractionalScore\": %f, \"feedback\": \"%s\"}' % (0.0, error))
        return

    #load normalization parameters
    try:
        scaler = pickle.load( open("saved/scaler.pkl", "rb"))
    except:
        error = 'Error loading scaler.pkl'
        print('{"fractionalScore\": %f, \"feedback\": \"%s\"}' % (0.0, error))
        return



    accurate_predictions, false_positives, missed_collisions = 0, 0, 0
    for action_i in range(total_actions):
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)

        network_param = get_network_param(sim_env, action, scaler)
        try:
            prediction = model(network_param)
        except:
            error = 'Error making inference.'
            print('{"fractionalScore\": %f, \"feedback\": \"%s\"}' % (0.0, error))
            return

        collision_predicted = prediction.item() > .5

        for action_timestep in range(action_repeat):
            _, collision, _ = sim_env.step(steering_force)
            if collision:
                steering_behavior.reset_action()
                break

        if (collision_predicted and collision):
            accurate_predictions += 1
            if previous_false_positive:
                false_positives -= 1
                accurate_predictions += 1
            previous_false_positive = False
        elif not collision_predicted and not collision:
            accurate_predictions += 1
            previous_false_positive = False
        elif collision_predicted and not collision:
            false_positives += 1
            previous_false_positive = True
        elif not collision_predicted and collision:
            missed_collisions += 1
            if previous_false_positive:
                false_positives -= 1
                accurate_predictions += 1
            previous_false_positive = False

    score = 1.0
    score = score - .03 * missed_collisions
    score = score - .01 * max(0, (false_positives - 10))
    if score < 0:
        score = 0
    feedback = 'False Positives: %d/%d  Missed Collisions %d/%d'%(false_positives, total_actions, missed_collisions, total_actions)
    print('{"fractionalScore\": %f, \"feedback\": \"%s\"}' % (score, feedback))


if __name__ == '__main__':
    total_actions = 100
    evaluate_accuracy(total_actions)

