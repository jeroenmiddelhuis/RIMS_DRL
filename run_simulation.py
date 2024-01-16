"""
.. include:: ../README.md
.. include:: example/example_arrivals/arrivals-example.md
.. include:: example/example_decision_mining/decision_mining-example.md
.. include:: example/example_process_times/process_times-example.md
"""

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


def setup(env: simpy.Environment, PATH_PETRINET, params, i, NAME):
    simulation_process = SimulationProcess(env, params)
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


def run_simulation(PATH_PETRINET, PATH_PARAMETERS, N_SIMULATION, N_TRACES, NAME):
    params = Parameters(PATH_PARAMETERS, N_TRACES, NAME)
    for i in range(0, N_SIMULATION):
        env = simpy.Environment()
        env.process(setup(env, PATH_PETRINET, params, i, NAME))
        env.run(until=params.SIM_TIME)

    #result = Result("output_{}".format(NAME), params)
    #result._analyse()

#-p example/RL_integration/bpi2012.pnml -s example/RL_integration/input.json -t 2 -i 1 -o RL_integration
PATH_PETRINET = 'example/RL_integration/bpi2012.pnml'
PATH_PARAMETERS = 'example/RL_integration/input.json'
N_TRACES = 5
OUTPUT = 'output_RL_integration'
NAME = 'RL_integration'
N_SIMULATION = 1

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_simulation(PATH_PETRINET, PATH_PARAMETERS, N_SIMULATION, N_TRACES, NAME)