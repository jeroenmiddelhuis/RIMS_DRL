from datetime import timedelta
import simpy
import pm4py
import random

from checking_process import SimulationProcess
from pm4py.objects.petri_net import semantics
from MAINparameters import*


class Token(object):

    def __init__(self, id, params, process: SimulationProcess, case, rp_feature='all_role'):
        self.id = id
        self.net, self.am, self.fm = pm4py.read_pnml(params.PATH_PETRINET)
        self.process = process
        self.start_time = params.START_SIMULATION
        self.pr_wip_initial = params.PR_WIP_INITIAL
        self.rp_feature = rp_feature
        self.params = params
        self.see_activity = False
        self.case = case
        self.pos = 0
        self.prefix = []

    def read_json(self, path):
        with open(path) as file:
            data = json.load(file)
            self.ac_index = data['ac_index']

    def simulation(self, env: simpy.Environment, writer, type, syn=False):
        trans = self.next_transition(syn)
        ### register trace in process ###
        resource_trace = self.process.get_resource_trace()
        resource_trace_request = resource_trace.request()

        while trans is not None:
            if self.see_activity:
                yield resource_trace_request
            if trans.label is not None:
                buffer = [self.id, trans.label]
                self.prefix.append(trans.label)
                ### call predictor for waiting time
                resource, resource_name = self.process.get_random_resource(trans.label)
                #if trans.label in self.params.ROLE_ACTIVITY:
                #    resource = self.process.get_resource(self.params.ROLE_ACTIVITY[trans.label])  ## definisco ruolo che deve eseguire (deve essere di tipo Resource
                #else:
                #    resource = self.process.get_resource('SYSTEM')
                transition = (self.params.INDEX_AC[trans.label], self.params.RESOURCE_TO_ROLE_LSTM[resource_name])
                self.prefix.append(trans.label)
                pr_wip_wait = self.pr_wip_initial + resource_trace.count
                #rp_oc = self.process.get_occupations_resource(resource.get_name())
                '''if type == 'rims_plus':
                    request_resource = resource.request()
                if type == 'rims_plus':
                    if len(resource.queue) > 0:
                        queue = len(resource.queue[-1])
                    else:
                        queue = 0
                else:
                    queue = -1'''
                #waiting = self.process.get_predict_waiting(str(self.id), pr_wip_wait, transition, rp_oc,self.start_time + timedelta(seconds=env.now), queue)
                #if self.see_activity:
                #    yield env.timeout(waiting)
                #if type == 'rims':
                ### register event in process ###
                resource_task = self.process.get_resource_event(trans.label)
                resource_task_request = resource_task.request()
                yield resource_task_request
                ### call predictor for processing time
                pr_wip = self.pr_wip_initial + resource_trace.count
                #rp_oc = self.process.get_occupations_resource(resource.get_name())
                rp_oc = self.process.get_occupations_resource(self.params.RESOURCE_TO_ROLE_LSTM[resource_name])
                initial_ac_wip = self.params.AC_WIP_INITIAL[trans.label] if trans.label in self.params.AC_WIP_INITIAL else 0
                ac_wip = initial_ac_wip + resource_task.count
                buffer.append(str(self.start_time + timedelta(seconds=env.now)))
                duration = self.process.get_predict_processing(str(self.id), pr_wip, transition, ac_wip, rp_oc, self.start_time + timedelta(seconds=env.now), -1)
                if self.params.POLICY == 'FIFO':
                    request_resource = resource.request()
                else:
                    request_resource = resource.request_SPT(duration) ### SPT
                yield request_resource
                if len(resource.queue) > 0:
                    queue = len(resource.queue[-1])
                else:
                    queue = 0
                buffer.append(str(self.start_time + timedelta(seconds=env.now)))
                #if trans.label == 'Start' or trans.label == 'End':
                #    yield env.timeout(0)
                #else:
                yield env.timeout(duration)
                buffer.append(str(self.start_time + timedelta(seconds=env.now)))
                buffer.append(resource.get_name())
                buffer.append(pr_wip_wait)
                buffer.append(ac_wip)
                buffer.append(duration)
                resource.release(request_resource)
                resource_task.release(resource_task_request)
                print(*buffer)
                writer.writerow(buffer)
                if trans.label != 'Start':
                    self.see_activity = True

            self.update_marking(trans)
            trans = self.next_transition(syn)

        resource_trace.release(resource_trace_request)

    def update_marking(self, trans):
        self.am = semantics.execute(trans, self.net, self.am)

    def next_transition(self, syn):
        all_enabled_trans = semantics.enabled_transitions(self.net, self.am)
        all_enabled_trans = list(all_enabled_trans)
        all_enabled_trans.sort(key=lambda x: x.name)
        label_element = str(list(self.am)[0])
        if len(all_enabled_trans) == 0:
            return None
        elif len(all_enabled_trans) > 1:
            if syn:
                prob = [0.50, 0.50]
                if label_element in ['exi_node_54ded9af-1e77-4081-8659-bd5554ae9b9d', 'exi_node_38c10378-0c54-4b13-8c4c-db3e4d952451', 'exi_node_52e223db-6ebf-4cc7-920e-9737fe97b655', 'exi_node_e55f8171-c3fc-4120-9ab1-167a472007b7']:
                    prob = [0.01, 0.99]
                elif label_element in ['exi_node_7af111c0-b232-4481-8fce-8d8278b1cb5a']:
                    prob = [0.99, 0.01]
                value = [*range(0, len(prob), 1)]
                next = int(random.choices(value, prob)[0])
            else:
                next = random.choices(list(range(0, len(all_enabled_trans), 1)))[0]
            return all_enabled_trans[next]
        else:
            return all_enabled_trans[0]
