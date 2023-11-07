import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
from typing import List



import csv
import simpy
import utility
from process import SimulationProcess
from event_trace import Token
from parameters import Parameters
import sys, getopt
from utility import *
import pm4py
from inter_trigger_timer import InterTriggerTimer
from result_analysis import Result
from datetime import timedelta
import warnings
import sys
import random
import json


#-p example/RL_integration/bpi2012.pnml -s example/RL_integration/input.json -t 2 -i 1 -o RL_integration
PATH_PETRINET = 'example/RL_integration/bpi2012.pnml'
PATH_PARAMETERS = 'example/RL_integration/input.json'
N_TRACES = 10
NAME = 'output_RL_integration'
N_SIMULATION = 1

DEBUG_PRINT = False

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    #run_simulation(PATH_PETRINET, PATH_PARAMETERS, N_SIMULATION, N_TRACES, NAME)

class gym_env(Env):
    def __init__(self) -> None:

                
        with open('./example/RL_integration/input.json', 'r') as f:
            input_data = json.load(f)

        self.resources = sorted(list(input_data["resource"].keys()))
        self.task_types = sorted(list(input_data["processing_time"].keys()))

        # Define the input and output of the agent (neural network)
        # The inputs are: 1) resource availability, 2) if a resource is busy, to what task_type, 3) number of tasks for each activity
        self.input = [resource + '_availability' for resource in self.resources] + \
                     [resource + '_to_task_type' for resource in self.resources] + \
                     [task_type for task_type in self.task_types]

        # The outputs are: the assignments (resource, task_type)
        self.output = [(resource, task_type) for task_type in self.task_types for resource in self.resources] + ['Postpone']

        # Observation space
        lows = np.array([0 for _ in range(len(self.input))])
        highs = np.array([1 for _ in range(len(self.input))])
        self.observation_space = spaces.Box(low=lows,
                                            high=highs,
                                            shape=(len(self.input),),
                                            dtype=np.float64)

        
        # Action space
        self.action_space = spaces.Discrete(len(self.output))

        self.completed_traces = []


        # TODO:
        # Please check if this is the correct way to create the simulation environment
        warnings.filterwarnings("ignore")
        params = Parameters(PATH_PARAMETERS, N_TRACES)
        self.env = simpy.Environment()
        self.simulation_process = SimulationProcess(self.env, params)
        self.env.process(self.setup(self.env, PATH_PETRINET, params, 1, NAME, self.simulation_process))


        # TODO:
        # Run simulation to first decision moment



    #!! The algorithm uses this function to interact with the environment
    # Every step is an observation (state, action, reward) which is used for training
    # The agent takes an action based on the current state, and the environment should transition to the next state
    # Take an action at every timestep and evaluate this action in the simulator
    def step(self, action):
        if DEBUG_PRINT:
            print('State, action:', self.state, action, self.output[action])
            print('Outpus', self.output)
            print('Mask:', self.action_masks())

        self.simulation_process.set_action_from_RL(self.output[action])

        # TODO:
        # Run the simulation to the next decision moment using the chosen action

        reward = 0
        # Gather rewards
        for trace_id, cycle_time in self.simulation_process.traces['ended']:
            if trace_id not in self.completed_traces:
                self.completed_traces.append(trace_id)
                reward += 1/(1+cycle_time)

        

        # TODO:
        # Add information of the simulation that says when an episode is finished (i.e., SIM_TIME is reached)
        # This flag, isTerminiated, is returned after every step
        # Gym automatically calls the reset() function to start a new episode
        isTerminated = False

        self.state = self.get_state()
        if DEBUG_PRINT: print('state after', self.state, '\n')
        return self.state, reward, isTerminated, {}, {}


    # Reset the environment -> restart simulation  
    def reset(self, seed=0):
        # TODO: 
        # Please have a look if this is a good way to reset the environment.
        # The simulation should restart and run until the first decision moment.

        params = Parameters(PATH_PARAMETERS, N_TRACES)
        self.env = simpy.Environment()
        self.simulation_process = SimulationProcess(self.env, params)
        self.env.process(self.setup(self.env, PATH_PETRINET, params, 1, NAME, self.simulation_process))

        self.state = self.get_state()
        return np.array(self.state), {}


    def get_state(self):
        env_state = self.simulation_process.get_state()
        if DEBUG_PRINT: print(self.env.now, self.simulation_process.get_state())
        resource_available = [1 if resource in env_state['resource_available'] else 0 for resource in self.resources]
        resource_assigned_to = [0 for _ in range(len(self.resources))]
        for (trace_id, task_type, res) in env_state['actual_assignment']:
            resource_assigned_to[self.resources.index(res)] = (self.task_types.index(task_type) + 1) / len(self.task_types)
        
        if len(self.simulation_process.tokens_pending) > 0:
            task_types_num = [min(1, sum([1 if token.next_activity == task_type else 0 for token in self.simulation_process.tokens_pending])/10) for task_type in self.task_types]
        else:
            task_types_num = [0 for _ in range(len(self.task_types))]

        return resource_available + resource_assigned_to + task_types_num


    # Create an action mask which invalidates ineligible actions
    def action_masks(self) -> List[bool]:
        mask = [0 for _ in range(len(self.output))]

        for resource in self.resources:
            for task_type in self.task_types:
                if self.state[self.input.index(resource + '_availability')] > 0 and self.state[self.input.index(task_type)] > 0:
                    mask[self.output.index(resource, task_type)] = 1
        mask[-1] = 1 # Postpone always possible

        return list(map(bool, mask))


    def setup(self, env: simpy.Environment, PATH_PETRINET, params, i, NAME, simulation_process):
        utility.define_folder_output("output/output_{}".format(NAME))
        f = open("output/output_{}/simulated_log_{}_{}".format(NAME, NAME, i)+".csv", 'w')
        writer = csv.writer(f)
        writer.writerow(Buffer(writer).get_buffer_keys())
        net, im, fm = pm4py.read_pnml(PATH_PETRINET)
        interval = InterTriggerTimer(params, simulation_process, params.START_SIMULATION)
        for i in range(0, params.TRACES):
            prefix = Prefix()
            itime = interval.get_next_arrival(env, i)
            yield env.timeout(itime)
            parallel_object = utility.ParallelObject()
            time_trace = env.now
            env.process(Token(i, net, im, params, simulation_process, prefix, 'sequential', writer, parallel_object, time_trace, None).simulation(env))
                

    def run_simulation(self, PATH_PETRINET, PATH_PARAMETERS, N_SIMULATION, N_TRACES, NAME):
        params = Parameters(PATH_PARAMETERS, N_TRACES)
        for i in range(0, N_SIMULATION):
            env = simpy.Environment()
            env.process(self.setup(env, PATH_PETRINET, params, i, NAME))
            env.run(until=params.SIM_TIME)

        result = Result("output_{}".format(NAME), params)
        result._analyse()
