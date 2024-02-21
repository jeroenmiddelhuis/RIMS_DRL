from datetime import datetime, timedelta
import simpy
import pm4py
import random
from process import SimulationProcess
from pm4py.objects.petri_net import semantics
from parameters import Parameters
from utility import Prefix
from simpy.events import AnyOf, AllOf, Event
import numpy as np
import copy
import csv
from utility import Buffer, ParallelObject
import custom_function as custom


class Token(object):

    def __init__(self, id: int, net: pm4py.objects.petri_net.obj.PetriNet, am: pm4py.objects.petri_net.obj.Marking, params: Parameters, process: SimulationProcess, prefix: Prefix, type: str,
                 writer: csv.writer, parallel_object: ParallelObject, time: float, env, calendar, values=None, print=False):
        self._id = id
        self._process = process
        self._start_time = params.START_SIMULATION
        self._start_trace = time
        self._params = params
        self._net = net
        self._am = am
        self._prefix = prefix
        self._type = type
        if type == 'sequential':
            self.see_activity = False
        else:
            self.see_activity = True
        self._writer = writer
        self._parallel_object = parallel_object
        self._buffer = Buffer(writer, values)
        self._buffer.set_feature("attribute_case", custom.case_function_attribute(self._id, time))
        self._trans = self.next_transition() ## next activity to perform for the trace
        self._next_activity = self._trans.label
        self._resource_trace = self._process._get_resource_trace()
        self.env = env
        self.pr_wip_initial = params.PR_WIP_INITIAL
        self.calendar = calendar
        self.END = False
        self.print = print

    def _delete_places(self, places):
        delete = []
        for place in places:
            for p in self._net.places:
                if str(place) in str(p.name):
                    delete.append(p)
        return delete

    def inter_trigger_time(self, time):
        yield self.env.timeout(time)
        ### set arrival time ####
        self._buffer.set_feature("id_case", self._id)
        self._buffer.set_feature("activity", "start")
        self._buffer.set_feature("start_time", self._start_time + timedelta(seconds=self.env.now))
        self._buffer.set_feature("end_time", self._start_time + timedelta(seconds=self.env.now))
        if self.print:
            self._buffer.print_values()
        time = self._start_time + timedelta(seconds=self.env.now)
        self._time_last_activity = self.env.now
        self._process.update_tokens_pending(self)
        self._process._update_state_traces(self._id, self.env)

    def update_time_after_postpone(self, time):
        yield self.env.timeout(time)
        self._process.update_kpi_trace(self._id, self.env.now)


    def simulation(self, action): ### action = {task: ... , resource: ....}
        """
            The main function to handle the simulation of a single trace
        """
        self._process.del_tokens_pending(self._id)
        start = self.env.now
        self._process.set_actual_assignment(self._id, action['task'], action['resource'])
        if self._prefix.is_empty(): ## no activities already executed
            self._resource_trace_request = self._resource_trace.request()

        self._buffer.reset()
        self._buffer.set_feature("id_case", self._id)
        self._buffer.set_feature("activity", action['task'])
        self._buffer.set_feature("prefix", self._prefix.get_prefix(self._start_time + timedelta(seconds=self.env.now)))
        self._buffer.set_feature("attribute_event", custom.event_function_attribute(self._id,
                                                                                    self._start_time + timedelta(
                                                              seconds=self.env.now)))
        self._buffer.set_feature("wip_wait", 0 if type != 'sequential' else self._resource_trace.count-1)
        self._buffer.set_feature("wip_wait", self._resource_trace.count)
        #waiting = self.define_waiting_time(action['task'])
        #if self._prefix.is_empty():
        #    yield env.timeout(waiting)

        ### define resource for activity
        resource = self._process._get_resource(action['resource'])
        self._buffer.set_feature("ro_single", self._process.get_occupations_single_role(resource._get_name()))

        self._buffer.set_feature("ro_total", self._process.get_occupations_all_role(self._params.RESOURCE_TO_ROLE_LSTM[action['resource']]))
        self._buffer.set_feature("role", resource._get_name())

        self._buffer.set_feature("ro_single", self._process.get_occupations_single_role(resource._get_name()))
        self._buffer.set_feature("ro_single", self._process.get_occupations_single_role(resource._get_name()))
        #self._buffer.set_feature("ro_total", self._process.get_occupations_all_role(self._params.RESOURCE_TO_ROLE_LSTM[action['resource']]))
        self._buffer.set_feature("role", resource._get_name())

        ### register event in process ###
        resource_task = self._process._get_resource_event(action['task'])
        self._buffer.set_feature("wip_activity", resource_task.count)

        queue = 0 if len(resource._queue) == 0 else len(resource._queue[-1])
        self._buffer.set_feature("queue", queue)
        self._buffer.set_feature("enabled_time", self._start_time + timedelta(seconds=self.env.now))

        request_resource = resource.request()
        yield request_resource

        #single_resource = self._process._set_single_resource(resource._get_name())
        self._buffer.set_feature("resource", resource._get_name())

        resource_task_request = resource_task.request()
        yield resource_task_request
        ### call predictor for processing time
        self._buffer.set_feature("wip_start",  self._resource_trace.count)
        self._buffer.set_feature("ro_single", self._process.get_occupations_single_role(resource._get_name()))
        #self._buffer.set_feature("ro_total", self._process.get_occupations_all_role(self._params.RESOURCE_TO_ROLE_LSTM[action['resource']]))
        self._buffer.set_feature("wip_activity", resource_task.count)

        if self.calendar:
            stop = resource.to_time_schedule(self._start_time + timedelta(seconds=self.env.now))
            yield self.env.timeout(stop)
        else:
            stop = 0
        self._buffer.set_feature("start_time", self._start_time + timedelta(seconds=self.env.now))

        #duration = self.define_processing_time(action['task'])
        #### Add prediction with LSTM model
        transition = (self._params.INDEX_AC[action['task']], self._params.RESOURCE_TO_ROLE_LSTM[action['resource']])
        ro_single = self._process.get_occupations_all_role(self._params.RESOURCE_TO_ROLE_LSTM[action['resource']])
        pr_wip = self.pr_wip_initial + self._resource_trace.count
        initial_ac_wip = self._params.AC_WIP_INITIAL[action['task']]
        ac_wip = initial_ac_wip + resource_task.count
        duration = self._process.get_predict_processing(str(self._id), pr_wip, transition,
                                                        ac_wip, ro_single, self._start_time + timedelta(seconds=self.env.now))
        #duration = self.call_custom_processing_time()
        yield self.env.timeout(duration)
        self._buffer.set_feature("wip_end", self._resource_trace.count)
        self._buffer.set_feature("end_time", self._start_time + timedelta(seconds=self.env.now))
        self._buffer.set_feature("queue", duration)
        self._prefix.add_activity(action['task'])
        if self.print:
            self._buffer.print_values()
        resource.release(request_resource)
        self._process._release_single_resource(self._id, resource._get_name(), action['task'])
        resource_task.release(resource_task_request)

        cycle_time = self.env.now - start - stop
        self._process.update_kpi_trace(self._id, self.env.now)

        self._update_marking(self._trans)
        self._trans = self.next_transition(self.env)
        if self._trans is not None:
            self._next_activity = self._trans.label
        else:
            self._next_activity = None
        self._time_last_activity = self.env.now
        if self._trans is None:  ### End of TRACE
            self._resource_trace.release(self._resource_trace_request)
            total_time = self.env.now - self._start_trace
            self._process._release_resource_trace(self._id, total_time, resource_task)
            self.END = True
        else:
            self._process.update_tokens_pending(self)

    def _get_resource_role(self, activity):
        elements = self._params.ROLE_ACTIVITY[activity.label]
        resource_object = []
        for e in elements:
            resource_object.append(self._process._get_resource(e, ''))
        return resource_object

    def _update_marking(self, trans):
        self._am = semantics.execute(trans, self._net, self._am)

    def _delete_tokens(self, name):
        to_delete = []
        for p in self._am:
            if p.name != name:
                to_delete.append(p)
        return to_delete

    def _check_probability(self, prob):
        """Check if the sum of probabilities is 1
        """
        if sum(prob) != 1:
            print('WARNING: The sum of the probabilities associated with the paths is not 1, to run the simulation we define equal probability')
            return False
        else:
            return True

    def _check_type_paths(self, prob):
        if type(prob[0]) is str:
            if sum([x == prob[0] for x in prob]) != len(prob):
                raise ValueError('ERROR: Not all path are defined as same type ', prob)
        elif type(prob[0]) is float:
            if sum([isinstance(x, float) for x in prob]) != len(prob):
                raise ValueError('ERROR: Not all path are defined as same type (float number) ', prob)
        else:
            raise ValueError("ERROR: Invalid input, specify the probability as AUTO, float number or CUSTOM ", prob)

    def _retrieve_check_paths(self, all_enabled_trans):
        prob = []
        for trans in all_enabled_trans:
            try:
                if trans.label:
                    prob.append(self._params.PROBABILITY[trans.label])
                else:
                    prob.append(self._params.PROBABILITY[trans.name])
            except:
                print('ERROR: Not all path probabilities are defined. Define all paths: ', all_enabled_trans)

        return prob

    def define_xor_next_activity(self, all_enabled_trans):
        """ Three different methods to decide which path following from XOR gateway:
        * Random choice: each path has equal probability to be chosen (AUTO)
        ```json
        "probability": {
            "A_ACCEPTED": "AUTO",
            "skip_2": "AUTO",
            "A_FINALIZED": "AUTO",
        }
        ```
        * Defined probability: in the file json it is possible to define for each path a specific probability (PROBABILITY as value)
        ```json
        "probability": {
            "A_PREACCEPTED": 0.20,
            "skip_1": 0.80
        }
        ```
        * Custom method: it is possible to define a dedicate method that given the possible paths it returns the one to
        follow, using whatever techniques the user prefers. (CUSTOM)
        ```json
        "probability": {
            "A_CANCELLED": "CUSTOM",
            "A_DECLINED": "CUSTOM",
            "tauSplit_5": "CUSTOM"
        }
        ```
        """
        prob = ['AUTO'] if not self._params.PROBABILITY else self._retrieve_check_paths(all_enabled_trans)
        self._check_type_paths(prob)
        if prob[0] == 'AUTO':
                next = random.choices(list(range(0, len(all_enabled_trans), 1)))[0]
        elif prob[0] == 'CUSTOM':
            next = self.call_custom_xor_function(all_enabled_trans)
        elif type(prob[0] == float()):
            if not self._check_probability(prob):
                value = [*range(0, len(prob), 1)]
                next = int(random.choices(value, prob)[0])
            else:
                next = random.choices(list(range(0, len(all_enabled_trans), 1)))[0]
        return all_enabled_trans[next]

    def define_processing_time(self, activity):
        """ Three different methods are available to define the processing time for each activity:
            * Distribution function: specify in the json file the distribution with the right parameters for each
            activity, see the [numpy_distribution](https://numpy.org/doc/stable/reference/random/generator.html) distribution, (DISTRIBUTION).
            **Be careful**: A negative value generated by the distribution is not valid for the simulator.
            ```json
             "processing_time": {
                 "A_FINALIZED": { "name": "uniform", "parameters": { "low": 3600, "high": 7200}},
             }
            ```
            * Custom method: it is possible to define a dedicated method that, given the activity and its
            characteristics, returns the duration of processing time required. (CUSTOM)
            ```json
            "processing_time": {
                 "A_FINALIZED":  { "name": "custom"}
            }
            ```
            * Mixed: It is possible to define a distribution function for some activities and a dedicated method for the others.
            ```json
            "processing_time": {
                 "A_FINALIZED":  { "name": "custom"},
                 "A_REGISTERED":  { "name": "uniform", "parameters": { "low": 3600, "high": 7200}}
            }
            ```
        """
        try:
            if self._params.PROCESSING_TIME[activity]["name"] == 'custom':
                duration = self.call_custom_processing_time()
            else:
                distribution = self._params.PROCESSING_TIME[activity]['name']
                parameters = self._params.PROCESSING_TIME[activity]['parameters']
                duration = getattr(np.random, distribution)(**parameters, size=1)[0]
                if duration < 0:
                    print("WARNING: Negative processing time",  duration)
                    duration = 0
        except:
            raise ValueError("ERROR: The processing time of", activity, "is not defined in json file")
        return duration

    def define_waiting_time(self, next_act):
        """ Three different methods are available to define the waiting time before each activity:
            * Distribution function: specify in the json file the distribution with the right parameters for each
            activity, see the [numpy_distribution](https://numpy.org/doc/stable/reference/random/generator.html) distribution, (DISTRIBUTION).
            **Be careful**: A negative value generated by the distribution is not valid for the simulator.
            ```json
             "waiting_time": {
                 "A_PARTLYSUBMITTED":  { "name": "uniform", "parameters": { "low": 3600, "high": 7200}}
             }
            ```
            * Custom method: it is possible to define a dedicated method that, given the next activity with its
            features, returns the duration of waiting time. (CUSTOM)
            ```json
            "waiting_time": {
                 "A_PARTLYSUBMITTED": { "name": "custom"}
            }
            ```
            * Mixed: As the processing time, it is possible to define a mix of methods for each activity.
            ```json
            "waiting_time": {
                 "A_PARTLYSUBMITTED":  { "name": "custom"},
                 "A_APPROVED":  { "name": "uniform", "parameters": { "low": 3600, "high": 7200}}
            }
            ```
        """
        try:
            if self._params.WAITING_TIME[next_act]["name"] == 'custom':
                duration = self.call_custom_waiting_time()
            else:
                distribution = self._params.WAITING_TIME[next_act]['name']
                parameters = self._params.WAITING_TIME[next_act]['parameters']
                duration = getattr(np.random, distribution)(**parameters, size=1)[0]
                if duration < 0:
                    print("WARNING: Negative waiting time",  duration)
                    duration = 0
        except:
            duration = 0

        return duration

    def call_custom_resource(self, trans, state):
        """
        Call to the custom functions in the file *custom_function.py*.
        """
        return custom.custom_resource(self._buffer, state, trans)

    def call_custom_processing_time(self):
        """
        Call to the custom functions in the file *custom_function.py*.
        """
        return custom.custom_processing_time(self._buffer)

    def call_custom_waiting_time(self):
        """
            Call to the custom functions in the file *custom_function.py*.
        """
        return custom.custom_waiting_time(self._buffer)

    def call_custom_xor_function(self, all_enabled_trans):
        """
            Call to the custom functions in the file *custom_function.py*.
        """
        return custom.custom_decision_mining(self._buffer)

    def next_transition(self, env= None):
        step = self.next_step()
        while step and step.label is None:
            self._update_marking(step)
            step = self.next_step()
        return step

    def next_step(self, env=None):
        """
        Method to define the next activity in the petrinet.
        """
        all_enabled_trans = semantics.enabled_transitions(self._net, self._am)
        all_enabled_trans = list(all_enabled_trans)
        all_enabled_trans.sort(key=lambda x: x.name)
        if len(all_enabled_trans) == 0:
            return None
        elif len(all_enabled_trans) == 1:
            return all_enabled_trans[0]
        else:
            return self.define_xor_next_activity(all_enabled_trans)
