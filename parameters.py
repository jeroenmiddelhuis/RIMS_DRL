"""
    Class for reading simulation parameters
"""
import json
import math
import os
from datetime import datetime


class Parameters(object):

    def __init__(self, path_parameters: str, traces: int, name_log: str, feature_role: str):
        self.TRACES = traces
        """TRACES: number of traces to generate"""
        self.PATH_PARAMETERS = path_parameters
        """PATH_PARAMETERS: path of json file for others parameters. """
        self.FEATURE_ROLE = feature_role
        if self.FEATURE_ROLE == 'all_role':
            self.prefix = ('_diapr', '_dpiapr', '_dwiapr')
        else:
            self.prefix = ('_dispr', '_dpispr', '_dwispr')
        self.NAME_EXP = name_log
        self.read_metadata_file()
        #self.parameters_LSTM_model()

    def read_metadata_file(self):
        '''
        Method to read parameters from json file, see *main page* to get the whole list of simulation parameters.
        '''
        if os.path.exists(self.PATH_PARAMETERS):
            with open(self.PATH_PARAMETERS) as file:
                data = json.load(file)
                roles_table = data['resource_table']
                self.START_SIMULATION = self._check_default_parameters(data, 'start_timestamp')
                self.SIM_TIME = self._check_default_parameters(data, 'duration_simulation')
                self.PROBABILITY = data['probability'] if 'probability' in data.keys() else []
                self.PROCESSING_TIME = data['processing_time']
                self.WAITING_TIME = data['waiting_time'] if 'waiting_time' in data.keys() else []
                self.INTER_TRIGGER = data["interTriggerTimer"]
                self.ROLE_ACTIVITY = dict()
                for elem in roles_table:
                    self.ROLE_ACTIVITY[elem['task']] = elem['role']

                if 'calendar' in data['interTriggerTimer'] and data['interTriggerTimer']['calendar']:
                    self.ROLE_CAPACITY = {'TRIGGER_TIMER': [math.inf, {'days': data['interTriggerTimer']['calendar']['days'], 'hour_min': data['interTriggerTimer']['calendar']['hour_min'], 'hour_max': data['interTriggerTimer']['calendar']['hour_max']}]}
                else:
                    self.ROLE_CAPACITY = {'TRIGGER_TIMER': [math.inf, []]}
                self._define_roles_resources(data['resource'])
        else:
            raise ValueError('Parameter file does not exist')

    def _define_roles_resources(self, roles):
        for idx, key in enumerate(roles):
            self.ROLE_CAPACITY[key] = [roles[key]['resources'], {'days': roles[key]['calendar']['days'],
                                                                      'hour_min': roles[key]['calendar']['hour_min'],
                                                                      'hour_max': roles[key]['calendar']['hour_max']}]

    def _check_default_parameters(self, data, type):
        if type == 'start_timestamp':
            value = datetime.strptime(data['start_timestamp'], '%Y-%m-%d %H:%M:%S') if type in data else datetime.now()
        elif type == 'duration_simulation':
            value = data['duration_simulation'] if type in data else 31536000
        return value

    def parameters_LSTM_model(self):
        self.MODEL_PATH_PROCESSING = './example/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[1] + '.h5'
        self.MODEL_PATH_WAITING = './example/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[2] + '.h5'
        self.METADATA = './example/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_meta.json'
        self.SCALER = './example/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_scaler.pkl'
        self.INTER_SCALER = './example/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_inter_scaler.pkl'
        self.END_INTER_SCALER = './example/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[
            0] + '_end_inter_scaler.pkl'

        if os.path.exists(self.METADATA):
            with open(self.METADATA) as file:
                data = json.load(file)
                self.INDEX_AC = data['ac_index']
                self.AC_WIP_INITIAL = data['inter_mean_states']['tasks']
                self.PR_WIP_INITIAL = round(data['inter_mean_states']['wip'])

                roles_table = data['roles_table']
                self.ROLE_ACTIVITY_LSTM = dict()
                for elem in roles_table:
                    self.ROLE_ACTIVITY_LSTM[elem['task']] = elem['role']

                self.INDEX_ROLE_LSTM = {'SYSTEM': 0}
                self.ROLE_CAPACITY_LSTM = {'SYSTEM': [1000, {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]}
                roles = data['roles']
                for idx, key in enumerate(roles):
                    self.INDEX_ROLE_LSTM[key] = idx
                    self.ROLE_CAPACITY_LSTM[key] = [len(roles[key]),
                                               {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]
                self.RESOURCE_TO_ROLE_LSTM = {}
                for idx, key in enumerate(roles):
                    for resource in roles[key]:
                        self.RESOURCE_TO_ROLE_LSTM[resource] = key