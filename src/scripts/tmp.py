import pandas as pd
import sys
sys.path.insert(0, '../modeling')
import visual as vis
sys.path.insert(0, '../utils')
import constants


test_config = ['g']
date = '1234'

ges_df = pd.read_csv(constants.RESULTS_PATH + '/ge/ge_files/ge_07022022-0243PM.csv',
                    header=None)
ges = ges_df.to_numpy()
train_configs = [str(i) for i in range(ges.shape[0])]

scores_df = pd.read_csv(constants.RESULTS_PATH + '/scores/score_files/scores_07022022-0243PM.csv',
                        header=None)
scores = scores_df.to_numpy()

vis.plot_ge_per_trainConfig(ges, scores, train_configs, test_config, date) 
