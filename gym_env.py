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
import warnings
import json
import math
import pm4py
from os.path import exists

CYCLE_TIME_MAX = 8.64e+6


DEBUG_PRINT = False
if __name__ == "__main__":
    warnings.filterwarnings("ignore")


class gym_env(Env):
    def __init__(self, name_log, POLICY=None, i=None) -> None:

        self.name_log = name_log
        self.policy = POLICY
        input_file = './example/' + self.name_log + '/input_' + self.name_log + '.json'
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        self.n_simulation = i

        self.resources = sorted(list(input_data["resource"].keys()))
        self.task_types = sorted(list(input_data["processing_time"].keys()))

        # Define the input and output of the agent (neural network)
        # The inputs are: 1) resource availability, 2) if a resource is busy, to what task_type, 3) number of tasks for each activity
        self.input = [resource + '_availability' for resource in self.resources] + \
                     [resource + '_to_task_type' for resource in self.resources] + \
                     [task_type for task_type in self.task_types]\

                     #+ ['wip', 'day', 'hour']

        # The outputs are: the assignments (resource, task_type)
        self.output = [(resource, task_type) for task_type in self.task_types for resource in self.resources] + ['Postpone']

        path_model = './example/' + self.name_log + '/' + self.name_log
        if exists(path_model + '_diapr_meta.json'):
            self.FEATURE_ROLE = 'all_role'
        elif exists(path_model + '_dispr_meta.json'):
            self.FEATURE_ROLE = 'no_all_role'
        else:
            self.FEATURE_ROLE = None
        self.PATH_PETRINET = './example/' + self.name_log + '/' + self.name_log + '.pnml'
        PATH_PARAMETERS = input_file
        self.N_TRACES = 2000
        self.CALENDAR = False ## If you want to use calendar or not
        self.PATH_LOG = './example/' + self.name_log + '/' + self.name_log + '.xes'
        self.params = Parameters(PATH_PARAMETERS, self.N_TRACES, self.name_log, self.FEATURE_ROLE)

        ### define possible assignments from log
        ###self.output = self.retrieve_possible_assignments(self.params.RESOURCE_TO_ROLE_LSTM)

        # Observation space
        lows = np.array([0 for _ in range(len(self.input))])
        highs = np.array([1 for _ in range(len(self.input))])
        self.observation_space = spaces.Box(low=lows,
                                            high=highs,
                                            shape=(len(self.input),),
                                            dtype=np.float64)

        
        # Action space
        self.action_space = spaces.Discrete(len(self.output))

        self.processes = []

        self.nr_steps = 0
        self.nr_postpone = 0

        warnings.filterwarnings("ignore")



    # Reset the environment -> restart simulation

    ## Read log to retrieve the possible assignments
    def retrieve_possible_assignments(self, resources):
        log = pm4py.read_xes(self.PATH_LOG)
        possible_assignment = set()
        for index, row in log.iterrows():
            if row['org:resource'] in resources.keys():
                possible_assignment.add((row['org:resource'], row['concept:name']))
        return list(possible_assignment)# + ['Postpone']

    def reset(self, seed=0):
        self.nr_steps = 0
        print('-------- Reseting environment --------')
        # TODO:
        # Please have a look if this is a good way to reset the environment.
        # The simulation should restart and run until the first decision moment.
        self.env = simpy.Environment()
        self.simulation_process = SimulationProcess(self.env, self.params)
        self.completed_traces = []
        utility.define_folder_output("output/output_{}".format(self.name_log))
        f = open("output/output_{}/simulated_log_{}_{}_{}".format(self.name_log, self.name_log, self.policy,str(self.n_simulation)) + ".csv", 'w')
        writer = csv.writer(f)
        writer.writerow(Buffer(writer).get_buffer_keys())
        net, im, fm = pm4py.read_pnml(self.PATH_PETRINET)
        interval = InterTriggerTimer(self.params, self.simulation_process, self.params.START_SIMULATION)
        self.tokens = {}
        prev = 0
        for i in range(0, self.N_TRACES):
            prefix = Prefix()
            itime = interval.get_next_arrival(self.env, i, self.name_log)
            prev = itime
            parallel_object = utility.ParallelObject()
            time_trace = self.env.now
            token = Token(i, net, im, self.params, self.simulation_process, prefix, 'sequential', writer, parallel_object,
                          itime, self.CALENDAR, None)
            self.tokens[i] = token
            #prev += itime
            self.env.process(token.inter_trigger_time(self.env, itime))
            
        not_token_ready = True
        while not_token_ready:
            if len(self.simulation_process.tokens_pending) > 0:
                not_token_ready = False
            self.env.step()
        self.state = self.get_state()
        return self.state, {}

    #!! The algorithm uses this function to interact with the environment
    # Every step is an observation (state, action, reward) which is used for training
    # The agent takes an action based on the current state, and the environment should transition to the next state
    # Take an action at every timestep and evaluate this action in the simulator
    def step(self, action):
        state = self.get_state()
        if self.output[action] == 'Postpone':
            self.nr_postpone += 1
        # print('State at start:', state, all(self.state == state))
        # print('Action:', action, self.output[action], self.env.now)
        self.nr_steps += 1
        if self.nr_steps % 2000 == 0:
            print('Steps:', self.nr_steps)
            print('State', self.get_state())
            print('Postpone actions:', self.nr_postpone, '/2000')
            #print('State simulator:', self.simulation_process.get_state())

            self.nr_postpone = 0

        trace_ongoing_prev = dict(self.simulation_process.get_state()['traces']['ongoing'])
        if self.output[action] != 'Postpone':
            token_id = None
            tokens_pending = {k: v for k, v in self.simulation_process.tokens_pending.items() if v[0]._next_activity == self.output[action][1]}
            if len(tokens_pending) > 1:
                token_id = max(tokens_pending.items(), key=lambda x: x[1][1])[0]
            else:
                token_id = list(tokens_pending.keys())[0]
            simulation = self.tokens[token_id].simulation(self.env, {'task': self.output[action][1],
                                                                     'resource': self.output[action][0]})
            self.env.process(simulation)
            self.next_decision_moment(self.output[action]) #arg not used
        else:
            self.next_decision_moment(self.output[action])

        ##### Reward at the end of an assignment
        '''
        trace_ongoing_actual = dict(self.simulation_process.get_state()['traces']['ongoing'])
        reward = 0
        for key in trace_ongoing_prev:
            if key in trace_ongoing_actual:
                reward += 1 - ((trace_ongoing_actual[key] - trace_ongoing_prev[key]) / CYCLE_TIME_MAX)
        for trace_id, cycle_time in self.simulation_process.traces['ended']:
            if trace_id not in self.completed_traces:
                reward += 1 - ((cycle_time - trace_ongoing_prev[trace_id]) / CYCLE_TIME_MAX)
                self.completed_traces.append(trace_id)
        '''
        # TODO:
        # Run the simulation to the next decision moment using the chosen action

        reward = 0
        # Gather rewards
        for trace_id, cycle_time in self.simulation_process.traces['ended']:
            if trace_id not in self.completed_traces:
                self.completed_traces.append(trace_id)
                reward += 1 / (1 + (cycle_time))

        # TODO:
        # Add information of the simulation that says when an episode is finished (i.e., SIM_TIME is reached)
        # This flag, isTerminated, is returned after every step
        # Gym automatically calls the reset() function to start a new episode
        if len(self.tokens) == 0:
            isTerminated = True
            print('Mean cycle time:', np.mean([cycle_time for (trace_id, cycle_time) in self.simulation_process.traces['ended']]))
            print('Total reward:', sum([1 / (1 + (cycle_time)) for (trace_id, cycle_time) in self.simulation_process.traces['ended']]))      
        else:
            isTerminated = False

        self.state = self.get_state()
        #print('State at end:', self.state, '\n')
        #if DEBUG_PRINT: print('state after', self.state, '\n')
        #print(self.simulation_process.traces)
        return self.state, reward, isTerminated, {}, {}

    def step_baseline(self, action):
        if action is not None:
            simulation = self.tokens[action[0]].simulation(self.env, {'task': action[1],
                                                                     'resource': action[2]})
            self.env.process(simulation)
        self.next_decision_moment(action)

        reward = 0
        # Gather rewards
        for trace_id, cycle_time in self.simulation_process.traces['ended']:
            if trace_id not in self.completed_traces:
                self.completed_traces.append(trace_id)
                reward += 1 / (1 + (cycle_time / 3600))

        if len(self.tokens) == 0:
            isTerminated = True
        else:
            isTerminated = False

        self.state = self.get_state()
        #if DEBUG_PRINT: print('state after', self.state, '\n')
        return self.state, reward, isTerminated, {}, {}


    def check_possible_assignments(self, resources_a, tokens):
        possible_tasks = []
        for action in self.output:
            if action[0] in resources_a:
                possible_tasks.append(action[1])
        possible_ass = False
        for token in tokens:
            if tokens[token][0]._next_activity in possible_tasks:
                possible_ass = True
        return possible_ass

    def delete_tokens_ended(self):
        delete = []
        for i in self.tokens:  ### update tokens ended
            if self.tokens[i].END is True:
                if self.tokens[i]._next_activity is None:
                    delete.append(i)
        for e in delete:
            del self.tokens[e]


    def next_decision_moment(self, action):
        next_step = True
        pre_state_res = set(self.simulation_process.get_state()['resource_available'])
        pre_state_tokens = len(self.simulation_process.get_state()['traces']['ongoing'])
        while next_step:
            if self.env.peek() == math.inf:
                next_step = False
            else:
                self.env.step()
            actual_state = self.simulation_process.get_state()
            state_res = set(actual_state['resource_available'])
            state_tokens = len(actual_state['traces']['ongoing'])
            ### exception case: all available resources and all tokens already arrived
            if (state_tokens+len(actual_state['traces']['ended'])) == self.params.TRACES and len(actual_state['resource_anvailable'])==0:
                next_step = False  ### todo better
            if pre_state_res != state_res or state_tokens > pre_state_tokens: ### state changed
                if self.check_possible_assignments(state_res, self.simulation_process.tokens_pending):
                    next_step = False
            pre_state_res = state_res
            pre_state_tokens = state_tokens
            self.delete_tokens_ended()

    def get_state(self):
        env_state = self.simulation_process.get_state()
        #if DEBUG_PRINT: print(self.env.now, self.simulation_process.get_state())
        resource_available = [1.0 if resource in env_state['resource_available'] else 0 for resource in self.resources]
        resource_assigned_to = [0.0 for _ in range(len(self.resources))]
        for (trace_id, task_type, res) in env_state['actual_assignment']:
            resource_assigned_to[self.resources.index(res)] = (self.task_types.index(task_type) + 1) / len(self.task_types)
        
        if len(self.simulation_process.tokens_pending) > 0:
            task_types_num = [min(1.0, sum([1 if self.simulation_process.tokens_pending[token][0]._next_activity == task_type else 0 for token in self.simulation_process.tokens_pending])/100) for task_type in self.task_types]
        else:
            task_types_num = [0.0 for _ in range(len(self.task_types))]
        #wip = [(len(env_state['traces']['ongoing'])+1)/1000]
        #time = [(env_state['time'].weekday() + 1)/7, (env_state['time'].hour + 1)/24]
        return np.array(resource_available + resource_assigned_to + task_types_num)

    # Create an action mask which invalidates ineligible actions
    def action_masks(self) -> List[bool]:
        mask = [0 for _ in range(len(self.output))]

        for resource in self.resources:
            for task_type in self.task_types:
                if self.state[self.input.index(resource + '_availability')] > 0 and self.state[self.input.index(task_type)] > 0:
                    if (resource, task_type) in self.output:
                        mask[self.output.index((resource, task_type))] = 1
        mask[-1] = 1 # Postpone always possible

        return list(map(bool, mask))

