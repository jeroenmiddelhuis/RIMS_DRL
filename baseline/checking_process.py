'''
classe process che contiene risorse ed esegue task
'''
from datetime import datetime
import simpy
from resource import Resource
from MAINparameters import*
import math
from call_LSTM import Predictor
import random
import pm4py

PATH_LOG = "/Users/francescameneghello/Desktop/RIMS/Logs/BPI_Challenge_2012_W_Two_TS.xes"


class SimulationProcess(object):

    def __init__(self, env: simpy.Environment, params):
        self.env = env
        self.params = params
        self.date_start = params.START_SIMULATION
        self.define_resources_for_activity()
        self.resource = self.define_single_resource()
        self.resource_events = self.define_resource_events(env)
        self.resource_trace = simpy.Resource(env, math.inf)
        self.buffer_traces = []
        self.predictor = Predictor((params.MODEL_PATH_PROCESSING, params.MODEL_PATH_WAITING), self.params)
        self.predictor.predict()

    def get_occupations_resource(self, role):
        occup = []
        if self.params.FEATURE_ROLE == 'all_role':
            for key in self.params.ROLE_CAPACITY_LSTM:
                if key != 'SYSTEM':
                    occup.append(self.get_occupations_single_role_LSTM(key))
        else:
            occup.append(self.get_occupations_single_role_LSTM(role))
        return occup

    def get_occupations_single_role_LSTM(self, role):
        """
        Method to retrieve the specified role occupancy in percentage, as an intercase feature:
        $\\frac{resources \: occupated \: in \:role}{total\:resources\:in\:role}$.
        """
        occup = 0
        for res in self.resource:
            if self.params.RESOURCE_TO_ROLE_LSTM[res] == role:
                occup += self.resource[res].get_resource().count
        occup = occup / self.params.ROLE_CAPACITY_LSTM[role][0]
        return round(occup, 2)

    def get_occupations_all_role(self):
        """
        Method to retrieve the occupancy in percentage of all roles, as an intercase feature.
        """
        list_occupations = []
        for res in self._resources:
            if res != 'TRIGGER_TIMER':
                occup = round(self._resources[res]._get_resource().count / self._resources[res]._capacity, 2)
                list_occupations.append(occup)
        return list_occupations


    def get_resource(self, resource_label):
        return self.resource[resource_label] ### ritorna tipo resource mio

    def get_resource_event(self, task):
        return self.resource_events[task]

    def get_resource_trace(self):
        return self.resource_trace

    def define_resource_events(self, env):
        resources = dict()
        for key in self.params.INDEX_AC:
            resources[key] = simpy.Resource(env, math.inf)
        return resources

    def get_predict_processing(self, cid, pr_wip, transition, ac_wip, rp_oc, time, queue):
        return self.predictor.processing_time(cid, pr_wip, transition, ac_wip, rp_oc, time, queue)

    def get_predict_waiting(self, cid, pr_wip, transition, rp_oc, time, queue):
        if queue < 0:
            return self.predictor.predict_waiting(cid, pr_wip, transition, rp_oc, time, queue)
        else:
            return self.predictor.predict_waiting_queue(cid, pr_wip, transition, rp_oc, time, queue)

    def get_random_resource(self, activity):
        available = self.get_resource_available()
        intersection = list(available.intersection(self.activity_resources[activity]))
        if len(intersection) > 0:
            res = random.choice(list(intersection))
        else:
            res = random.choice(list(self.activity_resources[activity]))
        return self.resource[res], res

    def define_resources_for_activity(self):
        log = pm4py.read_xes(PATH_LOG)
        self.activity_resources = {}
        self.resources_log = set()
        for index, row in log.iterrows():
            if row['org:resource'] in self.params.RESOURCE_TO_ROLE_LSTM.keys():
                if row['concept:name'] in self.activity_resources:
                    self.activity_resources[row['concept:name']].add(row['org:resource'])
                else:
                    self.activity_resources[row['concept:name']] = {row['org:resource']}
                self.resources_log.add(row['org:resource'])

    def define_single_resource(self):
        dict_res = dict()
        for res in self.resources_log:
            res_simpy = Resource(self.env, res, 1, {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}, self.date_start, self.params.POLICY)
            dict_res[res] = res_simpy
        return dict_res

    def get_resource_available(self):
        available = set()
        for res in self.resource:
            if res != 'TRIGGER_TIMER':
                occup = self.get_resource(res).resource_simpy.count
                if occup == 0:
                    available.add(res)
        return available