'''
Principal parameters to run the process

DA AGGIUNGERE: configurazione risorse, tempi per ogni attivita'
'''
import json
import os
from datetime import datetime

import pandas as pd


class Parameters(object):

    def __init__(self, name_exp, feature_role, iterations, type, policy):
        self.NAME_EXP = name_exp
        self.FEATURE_ROLE = feature_role
        self.PATH_PETRINET = self.NAME_EXP + '/' + self.NAME_EXP + '.pnml'
        if self.FEATURE_ROLE == 'all_role':
            self.prefix = ('_diapr', '_dpiapr', '_dwiapr')
        else:
            self.prefix = ('_dispr', '_dpispr', '_dwispr')

        self.MODEL_PATH_PROCESSING = self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[1] +'.h5'
        self.MODEL_PATH_WAITING = self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[2] + '.h5'

        self.SIM_TIME = 1460*36000000000000000  # 10 day
        #self.ARRIVALS = pd.read_csv(self.NAME_EXP + '/arrivals/iarr' + str(iterations) + '.csv', sep=',')
        self.ARRIVALS = pd.read_csv('/Users/francescameneghello/Documents/GitHub/rl-rims/example/BPI_Challenge_2012_W_Two_TS/inter_arrival_rate.csv')
        self.START_SIMULATION = datetime.strptime(self.ARRIVALS.loc[0].at["timestamp"], '%Y-%m-%d %H:%M:%S')
        self.N_TRACE = len(self.ARRIVALS)
        self.METADATA = self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_meta.json'
        self.SCALER = self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_scaler.pkl'
        self.INTER_SCALER = self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_inter_scaler.pkl'
        self.END_INTER_SCALER = self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_end_inter_scaler.pkl'
        self.POLICY = policy
        self.read_metadata_file()

    def read_metadata_file(self):
        if os.path.exists(self.METADATA):
            with open(self.METADATA) as file:
                data = json.load(file)
                self.INDEX_AC = data['ac_index']
                self.AC_WIP_INITIAL = data['inter_mean_states']['tasks']
                self.PR_WIP_INITIAL = round(data['inter_mean_states']['wip'])

                roles_table = data['roles_table']
                self.ROLE_ACTIVITY = dict()
                for elem in roles_table:
                    self.ROLE_ACTIVITY[elem['task']] = elem['role']

                self.INDEX_ROLE = {'SYSTEM': 0}
                self.ROLE_CAPACITY_LSTM = {'SYSTEM': [1000, {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]}
                roles = data['roles']
                for idx, key in enumerate(roles):
                    self.INDEX_ROLE[key] = idx
                    self.ROLE_CAPACITY_LSTM[key] = [len(roles[key]), {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]
                self.RESOURCE_TO_ROLE_LSTM = {}
                for idx, key in enumerate(roles):
                    for resource in roles[key]:
                        self.RESOURCE_TO_ROLE_LSTM[resource] = key