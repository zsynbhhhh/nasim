import pandas as pd
import matplotlib.pyplot as plt
def smooth4(csv_path,weight=0.9): #weight是平滑度，tensorboard 默认0.6
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Episode','Return','Step'],dtype={'Episode': int, 'Return': float, 'Step':int})
    return_csv = data['Return'].values
    step_csv = data['Step'].values
    last = return_csv[0]
    last_step = step_csv[0]
    smoothed = []
    smoothed_step = []
    for point in return_csv:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    for step in step_csv:
        smoothed_val_step = last_step * weight + (1 - weight) * step
        smoothed_step.append(smoothed_val_step)
        last_step = smoothed_val_step
    save = pd.DataFrame({'Episode':data['Episode'].values, 'Return':smoothed, 'Step':smoothed_step})
    save.to_csv(r'D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc4\smooth_ppo_sc4_return.csv')
def smooth1(csv_path,weight=0.9): #weight是平滑度，tensorboard 默认0.6
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['step','value'],dtype={'step':int,'value':float})
    scalar = data['value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'step':data['step'].values,'value':smoothed})
    save.to_csv(r'D:\Nasim-Zsy-ppo\nasim\nasim\train_data\ppo2\mediumgen3\smooth_DQN_mediumgen3_episode_return.csv')
# smooth1(r'D:\Nasim-Zsy-ppo\nasim\nasim\train_data\ppo2\large\NDSPI-large-episode-return.csv')
if __name__ == '__main__':
    smooth4(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc4\ppo_sc4_return.csv")
    # smooth1(r'D:\Nasim-Zsy-ppo\nasim\nasim\train_data\ppo2\mediumgen3\mediumgen3-episode-return.csv')
    pocp1 = pd.read_csv(
        r'D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc4\smooth_ppo_sc4_return.csv',
        usecols=['Episode', 'Return'])
    plt.plot(pocp1.Episode, pocp1.Return, lw=0.5, label='PPO-PTP', color='black', linestyle='-')

    pocp2 = pd.read_csv(r'D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc4\smooth_DQN_hugegen1_episode_return.csv', usecols=['Step', 'Value'])
    plt.plot(pocp2.Step, pocp2.Value, lw=0.5, label='DQN', color='black', linestyle=':')

    pocp3 = pd.read_csv(r'D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc4\smooth_NDSPI_hugegen1_episode_return.csv',
                        usecols=['Step', 'Value'])
    plt.plot(pocp3.Step, pocp3.Value, lw=0.5, label='NDSPI', color='black', linestyle='--')
    plt.xlabel('Episodes Scenario4')
    plt.ylabel('Reward')
    plt.subplots_adjust(left=0.15)
    plt.legend(loc=4)
    plt.gcf().set_size_inches(7, 4)
    plt.savefig(r'D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc4\scenario4_compare.png',dpi=1500)
    plt.show()