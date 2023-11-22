# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:16:15 2019

@author: Manuel Camargo
"""
import json
import math
import numpy as np


START_TIMEFORMAT = ''
INDEX_AC = None
INDEX_RL = None
DIM = dict()
TBTW = dict()
EXP = dict()



def predict_suffix_full(path_parameters, prefix, model):
    """Main function of the suffix prediction module.
    Args:
        timeformat (str): event-log date-time format.
        parameters (dict): parameters used in the training step.
        is_single_exec (boolean): generate measurments stand alone or share
                    results with other runing experiments (optional)
    """
    global INDEX_AC
    global INDEX_RL
    global DIM
    global TBTW
    global EXP


    # Loading of parameters from training
    with open(path_parameters) as file:
        data = json.load(file)
        EXP = {k: v for k, v in data['exp_desc'].items()}
        DIM['samples'] = int(data['dim']['samples'])
        DIM['time_dim'] = int(data['dim']['time_dim'])
        DIM['features'] = int(data['dim']['features'])
        TBTW['max_tbtw'] = float(data['max_tbtw'])
        INDEX_AC = {int(k): v for k, v in data['index_ac'].items()}
        INDEX_RL = {int(k): v for k, v in data['index_rl'].items()}
        file.close()


    #   Next event selection method and numbers of repetitions
    variants = [{'imp': 'Random Choice', 'rep': 1},
                {'imp': 'Arg Max', 'rep': 1}]
    #   Generation of predictions
    time = predict(model, prefix, 'Arg Max')

    return time


def predict(model, prefixes, imp):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        imp (str): method of next event selection.
    """
    # Generation of predictions
    for prefix in prefixes:
        #print('INPUT', prefix)
        # Activities and roles input shape(1,5)
        x_ac_ngram = np.append(
            np.zeros(DIM['time_dim']),
            np.array(prefix['ac_pref']),
            axis=0)[-DIM['time_dim']:].reshape((1, DIM['time_dim']))

        x_rl_ngram = np.append(
            np.zeros(DIM['time_dim']),
            np.array(prefix['rl_pref']),
            axis=0)[-DIM['time_dim']:].reshape((1, DIM['time_dim']))

        # times input shape(1,5,1)
        x_t_ngram = np.array([np.append(
            np.zeros(DIM['time_dim']),
            np.array(prefix['t_pref']),
            axis=0)[-DIM['time_dim']:].reshape((DIM['time_dim'], 1))])
        acum_tbtw = 0
        ac_suf, rl_suf = list(), list()
        for _ in range(1, 2):
            predictions = model.predict([x_ac_ngram, x_rl_ngram, x_t_ngram], verbose=0)
            if imp == 'Random Choice':
                # Use this to get a random choice following as PDF the predictions
                pos = np.random.choice(np.arange(0, len(predictions[0][0])), p=predictions[0][0])
                pos1 = np.random.choice(np.arange(0, len(predictions[1][0])), p=predictions[1][0])
            elif imp == 'Arg Max':
                # Use this to get the max prediction
                pos = np.argmax(predictions[0][0])
                pos1 = np.argmax(predictions[1][0])
            # Activities accuracy evaluation
            x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
            x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
            x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
            x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
            x_t_ngram = np.append(x_t_ngram, [predictions[2]], axis=1)
            x_t_ngram = np.delete(x_t_ngram, 0, 1)
            # Stop if the next prediction is the end of the trace
            # otherwise until the defined max_size
            ac_suf.append(pos)
            rl_suf.append(pos1)
            if EXP['norm_method'] == 'lognorm':
                acum_tbtw += math.expm1(predictions[2][0][0] * TBTW['max_tbtw'])
            else:
                acum_tbtw += np.rint(predictions[2][0][0] * TBTW['max_tbtw'])
            if INDEX_AC[pos] == 'end':
                break
        #print('PREDICT--> ACTIVITY', ac_suf, 'ROLE', rl_suf, 'TIME', acum_tbtw)
    return acum_tbtw

