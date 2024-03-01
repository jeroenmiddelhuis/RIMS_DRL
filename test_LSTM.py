
from parameters import Parameters
from os.path import exists
from call_LSTM import Predictor
from datetime import datetime, timedelta


name_log = 'ConsultaDataMining201618'
PATH_PARAMETERS = input_file = './example/' + name_log + '/input_' + name_log + '.json'
N_TRACES = 1
path_model = './example/' + name_log + '/' + name_log
if exists(path_model + '_diapr_meta.json'):
    FEATURE_ROLE = 'all_role'
else:
    FEATURE_ROLE = 'no_all_role'

params = Parameters(PATH_PARAMETERS, N_TRACES, name_log, FEATURE_ROLE)
predictor = Predictor((params.MODEL_PATH_PROCESSING, params.MODEL_PATH_WAITING), params)
predictor.predict()

print('LOG', name_log)

duration = predictor.processing_time(0, 43, (1, 'Role 2'), 1.0, [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], datetime.strptime('2012-02-14 09:00:00', '%Y-%m-%d %H:%M:%S'))
print('ACTIVITY : Traer informacion estudiante', duration)

duration = predictor.processing_time(0, 43, (2, 'Role 2'), 1.1023185483870968, [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], datetime.strptime('2012-02-14 09:00:55', '%Y-%m-%d %H:%M:%S'))
print('ACTIVITY : Radicar Solicitud Homologacion', duration)

duration = predictor.processing_time(0, 43, (5, 'Role 2'), 1.2653729838709677, [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], datetime.strptime('2012-02-14 09:12:38', '%Y-%m-%d %H:%M:%S'))
print('ACTIVITY : Cancelar Solicitud', duration)

duration = predictor.processing_time(0, 43, (6, 'Role 2'), 1.0, [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], datetime.strptime('2012-02-14 09:12:38', '%Y-%m-%d %H:%M:%S'))
print('ACTIVITY : Notificacion estudiante cancelacion soli', duration)





name_log = 'BPI_Challenge_2012_W_Two_TS'
PATH_PARAMETERS = input_file = './example/' + name_log + '/input_' + name_log + '.json'
N_TRACES = 1
path_model = './example/' + name_log + '/' + name_log
if exists(path_model + '_diapr_meta.json'):
    FEATURE_ROLE = 'all_role'
else:
    FEATURE_ROLE = 'no_all_role'

params = Parameters(PATH_PARAMETERS, N_TRACES, name_log, FEATURE_ROLE)
predictor = Predictor((params.MODEL_PATH_PROCESSING, params.MODEL_PATH_WAITING), params)
predictor.predict()

print('LOG', name_log)

duration = predictor.processing_time(0, 290, (4, 'Role 6'), 1.001501168300547, [1.0], datetime.strptime('2012-02-14 09:00:00', '%Y-%m-%d %H:%M:%S'))
print('ACTIVITY : W_Beoordelen fraude', duration)

duration = predictor.processing_time(0, 290, (1, 'Role 1'), 1.517955278238281, [0.17], datetime.strptime('2012-02-14 09:02:46', '%Y-%m-%d %H:%M:%S'))
print('ACTIVITY : W_Afhandelen leads', duration)
