from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation_Submit import Simulation
from generator_Submit import TrafficGenerator
from memory_Submit import Memory
from model_Submit import model_to_train
from visualization_Submit import Visualization
from utils_Submit import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings_Submit.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    Model = model_to_train(
        config['total_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['alpha'], 
        layer_states=config['total_states'], 
        layer_actions=config['total_actions']
    )

    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGenerator = TrafficGenerator(
        config['max_steps'], 
        config['total_cars']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        Memory,
        TrafficGenerator,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['total_states'],
        config['total_actions'],
        config['epoch']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n episode', str(episode+1), '/', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n Start time:", timestamp_start)
    print(" End time:", datetime.datetime.now())
    print("Duration: ",(datetime.datetime.now()-timestamp_start))
    print(" Session info saved at:", path)

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')

    Model.save_model(path)

    copyfile(src='training_settings_Submit.ini', dst=os.path.join(path, 'training_settings_Submit.ini'))

    #Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    #Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    #Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')