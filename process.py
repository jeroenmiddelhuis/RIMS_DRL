'''
Class to manage the resources shared by all the traces in the process.

<img src="../docs/images/process_class.png" alt="Alt Text" width="780">
'''
import simpy
from role_simulator import RoleSimulator
import math
from parameters import Parameters
import custom_function as custom
from simpy.events import AnyOf

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
        #self.tokens_event = {'None': self._env.event()}



    #def start_token(self, token_id, activity, resource):
    #    ### in base a cosa mi dice STEP LIBERO LA RISORSA DEL TOKEN
    #    self.token_handle[token_id]['res'] = resource
    #    self.token_handle[token_id]['obj'].release(self.token_handle[token_id]['request'])
    #    print('RELEASE', token_id)

    #def get_resource_RL(self, id):
    #    return self._get_resource(self.token_handle[id]['res'])

    #def get_token(self, id):
    #    return self.token_handle[id]['obj']

    def update_tokens_pending(self, token):
        self.tokens_pending[token._id] = token

    def del_tokens_pending(self, id):
        del self.tokens_pending[id]

    def define_single_role(self):
        """
        Definition of a *RoleSimulator* object for each role in the process.
        """
        set_resource = list(self._params.ROLE_CAPACITY.keys())
        dict_role = dict()
        for res in set_resource:
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

    '''def get_state(self):
        state = dict()
        for res in self._resources:
            if res != 'TRIGGER_TIMER':
                occup = self._resources[res]._get_resource().count
                no_occup = self._resources[res]._capacity - self._resources[res]._get_resource().count
                state[res] = {'not_available': occup, 'available': no_occup, 'resource_available': self._resources[res].get_available_single_resource(),
                              'resource_anavailable': self._resources[res].get_anavailable_single_resource()}
        state['actual_assignment'] = self._actual_assignment
        state['traces'] = self.traces
        return state

    def _get_resource(self, resource_label, activity):
        if resource_label != 'TRIGGER_TIMER':
            self._actual_assignment.append((activity, resource_label))
        self.check_resource_available()
        return self._resources[resource_label]'''

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

    def _release_resource_trace(self, id, time):
        tupla = ()
        for i in self.traces["ongoing"]:
            if i[0] == id:
                tupla = i
        self.traces["ongoing"].remove(tupla)
        self.traces["ended"].append((id, time))

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
