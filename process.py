'''
Class to manage the resources shared by all the traces in the process.

<img src="../docs/images/process_class.png" alt="Alt Text" width="780">
'''
import simpy
from role_simulator import RoleSimulator
import math
from parameters import Parameters
from datetime import datetime, timedelta
from call_LSTM import Predictor


class SimulationProcess(object):

    def __init__(self, env: simpy.Environment, params: Parameters):
        self._env = env
        self._params = params
        self._date_start = params.START_SIMULATION
        self._resources = self.define_single_role()
        self._resource_events = self._define_resource_events(env)
        self._resource_trace = simpy.Resource(env, math.inf)
        self._am_parallel = []
        self._actual_assignment = []
        self.traces = {"ongoing": [], "ended": []}
        self.resources_available = True ### check if there are resource to assign one event
        self.tokens_pending = {} ### dictionary keyv = id_case, element = tokenOBJ
        self.next_assign = None

        self.predictor = Predictor((self._params.MODEL_PATH_PROCESSING, self._params.MODEL_PATH_WAITING), self._params)
        self.predictor.predict()


    def update_tokens_pending(self, token):
        self.tokens_pending[token._id] = [token, token._time_last_activity]

    def del_tokens_pending(self, id):
        del self.tokens_pending[id]

    def define_single_role(self):
        """
        Definition of a *RoleSimulator* object for each role in the process.
        """
        set_resource = list(self._params.ROLE_CAPACITY.keys())
        dict_role = dict()
        for res in set_resource:
            #if res in self._params.RESOURCE_TO_ROLE_LSTM.keys():
            res_simpy = RoleSimulator(self._env, res, self._params.ROLE_CAPACITY[res][0],
                                      self._params.ROLE_CAPACITY[res][1])
            dict_role[res] = res_simpy
        return dict_role

    def get_occupations_single_role(self, resource):
        """
        Method to retrieve the specified role occupancy in percentage, as an intercase feature:
        $\\frac{resources \: occupated \: in \:role}{total\:resources\:in\:role}$.
        """
        occup = self._resources[resource]._get_resource().count / self._resources[resource]._capacity
        return round(occup, 2)

    def get_occupations_all_role(self, role):
        """
        Method to retrieve the occupancy in percentage of all roles, as an intercase feature.
        """
        occup = []
        if self._params.FEATURE_ROLE == 'all_role':
            for key in self._params.ROLE_CAPACITY_LSTM:
                if key != 'SYSTEM' and key != 'TRIGGER_TIMER':
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
        for res in self._resources:
            if res != 'TRIGGER_TIMER' and  self._params.RESOURCE_TO_ROLE_LSTM[res] == role:
                occup += self._resources[res]._get_resource().count
        occup = occup / self._params.ROLE_CAPACITY_LSTM[role][0]
        return round(occup, 2)


    def get_state(self):
        state = {'resource_available': [], 'resource_anvailable': []}
        for res in self._resources:
            if res != 'TRIGGER_TIMER':
                occup = self._resources[res]._get_resource().count
                if occup == 1:
                    state['resource_anvailable'].append(res)
                else:
                    state['resource_available'].append(res)

        state['actual_assignment'] = self._actual_assignment
        state['traces'] = self.traces
        return state

    def _get_resource(self, res):
        return self._resources[res]
    def set_actual_assignment(self, id, activity, res):
        self._actual_assignment.append((id, activity, res))

    def _get_resource_event(self, task):
        return self._resource_events[task]

    def _get_resource_trace(self):
        return self._resource_trace

    def _update_state_traces(self, id, env):
        self.traces["ongoing"].append((id, env.now))

    def _release_resource_trace(self, id, time, request_resource):
        tupla = ()
        for i in self.traces["ongoing"]:
            if i[0] == id:
                tupla = i
        self.traces["ongoing"].remove(tupla)
        self.traces["ended"].append((id, time))
        self._resource_trace.release(request_resource)

    def update_kpi_trace(self, id, time):
        tupla = ()
        for i in self.traces["ongoing"]:
            if i[0] == id:
                tupla = i
        self.traces["ongoing"].remove(tupla)
        self.traces["ongoing"].append((id, time))

    def _define_resource_events(self, env):
        resources = dict()
        for key in self._params.PROCESSING_TIME.keys():
            resources[key] = simpy.Resource(env, math.inf)
        return resources

    def _set_single_resource(self, resource_task):
        return self._resources[resource_task]._get_resources_name()

    def _release_single_resource(self, id, res, activity):
       if res != 'TRIGGER_TIMER':
            self._actual_assignment.remove((id, activity, res))

    def get_predict_processing(self, cid, pr_wip, transition, ac_wip, rp_oc, time):
        return self.predictor.processing_time(cid, pr_wip, transition, ac_wip, rp_oc, time)