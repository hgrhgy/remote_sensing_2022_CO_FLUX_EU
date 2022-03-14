import glob
import pickle
import matplotlib.pyplot as plt

import numpy as np

default_fontdict = {
    'fontsize': 18,
    'family': 'Times New Roman'
}
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['means'], marker='o', markeredgecolor=color, markerfacecolor=color)


if __name__ == '__main__':
    mop_data_2017 = 'mop_data_2017'
    mop_data_2018 = 'mop_data_2018'
    mop_data_2019 = 'mop_data_2019'
    mop_data_2020 = 'mop_data_2020'
    da_data_2017 = 'da_data_2017_new_sim'
    da_data_2018 = 'da_data_2018_new_sim'
    da_data_2019 = 'da_data_2019_new_sim'
    da_data_2020 = 'da_data_2020_new_sim'
    # da_data_2017 = 'post_mean_6h_2017_new_sim'
    # da_data_2018 = 'post_mean_6h_2018_new_sim'
    # da_data_2019 = 'post_mean_6h_2019_new_sim'
    # da_data_2020 = 'post_mean_6h_2020_new_sim'

    obs_data_2017 = 'obs_data_2017'
    obs_data_2018 = 'obs_data_2018'
    obs_data_2019 = 'obs_data_2019'
    obs_data_2020 = 'obs_data_2020'
    # sim_data_2017 = 'sim_data_2017'
    # sim_data_2018 = 'sim_data_2018'
    # sim_data_2019 = 'sim_data_2019'
    # sim_data_2020 = 'sim_data_2020'
    sim_data_2017 = 'sim_data_2017_new_sim'
    sim_data_2018 = 'sim_data_2018_new_sim'
    sim_data_2019 = 'sim_data_2019_new_sim'
    sim_data_2020 = 'sim_data_2020_new_sim'
    init_data_2017 = 'init_data_2017'
    init_data_2018 = 'init_data_2018'
    init_data_2019 = 'init_data_2019'
    init_data_2020 = 'init_data_2020'

    with open(mop_data_2017, 'rb') as fp:
        boxdata_mop_2017 = pickle.load(fp)
    with open(sim_data_2017, 'rb') as fp:
        boxdata_sim_2017 = pickle.load(fp)
    with open(da_data_2017, 'rb') as fp:
        boxdata_da_2017 = pickle.load(fp)
    with open(obs_data_2017, 'rb') as fp:
        boxdata_obs_2017 = pickle.load(fp)
    with open(init_data_2017, 'rb') as fp:
        boxdata_init_2017 = pickle.load(fp)
        boxdata_init_2017 = [v for v in boxdata_init_2017 if not np.isnan(v)]

    with open(mop_data_2018, 'rb') as fp:
        boxdata_mop_2018 = pickle.load(fp)
    with open(sim_data_2018, 'rb') as fp:
        boxdata_sim_2018 = pickle.load(fp)
    with open(da_data_2018, 'rb') as fp:
        boxdata_da_2018 = pickle.load(fp)
    with open(obs_data_2018, 'rb') as fp:
        boxdata_obs_2018 = pickle.load(fp)
    with open(init_data_2018, 'rb') as fp:
        boxdata_init_2018 = pickle.load(fp)
        boxdata_init_2018 = [v for v in boxdata_init_2018 if not np.isnan(v)]

    with open(mop_data_2019, 'rb') as fp:
        boxdata_mop_2019 = pickle.load(fp)
    with open(sim_data_2019, 'rb') as fp:
        boxdata_sim_2019 = pickle.load(fp)
    with open(da_data_2019, 'rb') as fp:
        boxdata_da_2019 = pickle.load(fp)
    with open(obs_data_2019, 'rb') as fp:
        boxdata_obs_2019 = pickle.load(fp)
    with open(init_data_2019, 'rb') as fp:
        boxdata_init_2019 = pickle.load(fp)
        boxdata_init_2019 = [v for v in boxdata_init_2019 if not np.isnan(v)]

    with open(init_data_2020, 'rb') as fp:
        boxdata_init_2020 = pickle.load(fp)
        boxdata_init_2020 = [v for v in boxdata_init_2020 if not np.isnan(v)]
    with open(mop_data_2020, 'rb') as fp:
        boxdata_mop_2020 = pickle.load(fp)
    with open(sim_data_2020, 'rb') as fp:
        boxdata_sim_2020 = pickle.load(fp)
    with open(da_data_2020, 'rb') as fp:
        boxdata_da_2020 = pickle.load(fp)
    with open(obs_data_2020, 'rb') as fp:
        boxdata_obs_2020 = pickle.load(fp)

    print('exp2017 median:', np.median(boxdata_init_2017), np.median(boxdata_sim_2017), np.median(boxdata_da_2017),
          np.median(boxdata_mop_2017), np.median(boxdata_obs_2017))
    print('exp2017 mean:', np.mean(boxdata_init_2017), np.mean(boxdata_sim_2017), np.mean(boxdata_da_2017),
          np.mean(boxdata_mop_2017), np.mean(boxdata_obs_2017))

    print('exp2018 median:', np.median(boxdata_init_2018), np.median(boxdata_sim_2018), np.median(boxdata_da_2018),
          np.median(boxdata_mop_2018), np.median(boxdata_obs_2018))
    print('exp2018 mean:', np.mean(boxdata_init_2018), np.mean(boxdata_sim_2018), np.mean(boxdata_da_2018),
          np.mean(boxdata_mop_2018), np.mean(boxdata_obs_2018))

    print('exp2019 median:', np.median(boxdata_init_2019), np.median(boxdata_sim_2019), np.median(boxdata_da_2019),
          np.median(boxdata_mop_2019), np.median(boxdata_obs_2019))
    print('exp2019 mean:', np.mean(boxdata_init_2019), np.mean(boxdata_sim_2019), np.mean(boxdata_da_2019),
          np.mean(boxdata_mop_2019), np.mean(boxdata_obs_2019))

    print('exp2020 median:', np.median(boxdata_init_2020), np.median(boxdata_sim_2020), np.median(boxdata_da_2020),
          np.median(boxdata_mop_2020), np.median(boxdata_obs_2020))
    print('exp2020 mean:', np.mean(boxdata_init_2020), np.mean(boxdata_sim_2020), np.mean(boxdata_da_2020),
          np.mean(boxdata_mop_2020), np.mean(boxdata_obs_2020))


    data_mop = [boxdata_mop_2017, boxdata_mop_2018, boxdata_mop_2019, boxdata_mop_2020]
    data_sim = [boxdata_sim_2017,boxdata_sim_2018, boxdata_sim_2019, boxdata_sim_2020]
    data_obs = [boxdata_obs_2017, boxdata_obs_2018, boxdata_obs_2019, boxdata_obs_2020]
    data_da = [boxdata_da_2017, boxdata_da_2018, boxdata_da_2019, boxdata_da_2020]
    data_init = [boxdata_init_2017, boxdata_init_2018, boxdata_init_2019, boxdata_init_2020]

    ticks = ['Exp2017', 'Exp2018', 'Exp2019', 'Exp2020']

    fig = plt.figure(figsize=(8, 6))

    bplot_init = plt.boxplot(data_init, positions=np.array(range(len(data_init))) * 4.0 - 1.2, sym='', widths=0.5,
                             whis=(10, 90), showmeans=True)
    bplot_sim = plt.boxplot(data_sim, positions=np.array(range(len(data_sim))) * 4.0 - 0.6, sym='', widths=0.5,
                            whis=(10, 90), showmeans=True)
    bplot_da = plt.boxplot(data_da, positions=np.array(range(len(data_da))) * 4.0, sym='', widths=0.5, whis=(10, 90),
                           showmeans=True)
    bplot_mop = plt.boxplot(data_mop, positions=np.array(range(len(data_mop))) * 4.0 + 0.6, sym='', widths=0.5,
                            whis=(10, 90), showmeans=True)
    bplot_obs = plt.boxplot(data_obs, positions=np.array(range(len(data_obs))) * 4.0 + 1.2, sym='', widths=0.5,
                            whis=(10, 90), showmeans=True)

    # for j in range(len(data_init)):
    #     idata = data_init[j]
    #     sdata = data_obs[j]
    #     idata = np.array(idata)
    #     sdata = np.array(sdata)
    #     bias_mean = np.mean(sdata - idata)
    #     mse = np.mean((sdata - idata) ** 2)
    #     rmse = mse ** 0.5
    #     rs = stats.pearsonr(sdata, idata)[0]
    #
    #     print('EXP_%d' %(2017+ j), 'mse', mse)
    #     print('EXP_%d' %(2017+ j), 'rmse', mse)
    #     print('EXP_%d' %(2017+ j), 'r2', mse)


    set_box_color(bplot_init, '#e6550d')
    set_box_color(bplot_sim, '#762a83')  # colors are from http://colorbrewer2.org/
    set_box_color(bplot_da, '#d73027')
    set_box_color(bplot_mop, '#2d2d2d')
    set_box_color(bplot_obs, '#1b7837')

    plt.plot([], c='#e6550d', label='CAM-Chem')
    plt.plot([], c='#762a83', label='SIM')
    plt.plot([], c='#d7191c', label='DA')
    plt.plot([], c='#2d2d2d', label='MOPITT')
    plt.plot([], c='#1b7837', label='ICOS')

    plt.legend(fontsize=16)

    plt.xticks(range(0, len(ticks) * 4, 4), ticks)
    plt.xlim(-2, len(ticks) * 4 - 2)
    plt.ylabel('CO Concentration (ppb)', fontdict=default_fontdict)
    plt.yticks(fontsize=16, family='Times New Roman')
    plt.xticks(fontsize=16, family='Times New Roman')
    # plt.title(
    #     'CO Assimilation Results Compared With Observations', fontdict=default_fontdict)

    plt.savefig(r'cc_boxplot_v3.png', dpi=300)
    plt.show()
    # # pd_2018 = pd.DataFrame({'MOPITT': boxdata_mop_2018, 'MODEL':boxdata_model_2018, 'SITE': boxdata_obs_2018})
    # # pd_2019 = pd.DataFrame({'MOPITT': boxdata_mop_2019, 'MODEL': boxdata_model_2019, 'SITE': boxdata_obs_2019})
    # # pd_2020 = pd.DataFrame({'MOPITT': boxdata_mop_2020, 'MODEL': boxdata_model_2020, 'SITE': boxdata_obs_2020})
    #
    # # sns.boxplot(x='2018', y='ppb', data=pd_2018)
    # # sns.boxplot(x='2019', y='ppb', data=pd_2019)
    # # sns.boxplot(x='2020', y='ppb', data=pd_2020)
    # bplot_2018 = plt.boxplot([boxdata_mop_2018, boxdata_model_2018, boxdata_obs_2018], positions=[1, 2, 3],
    #                          widths=0.6, showmeans=True,
    #                          patch_artist=True, meanline=True,
    #                          labels=['MOPITT', 'MODEL', 'SITE'], whis=[10, 90])
    # bplot_2019 = plt.boxplot([boxdata_mop_2018, boxdata_model_2018, boxdata_obs_2018], positions=[4, 5, 6],
    #                          widths=0.6, showmeans=True,
    #                          patch_artist=True,
    #                          meanline=True,
    #                          labels=['MOPITT', 'MODEL', 'SITE'], whis=[10, 90])
    # bplot_2020 = plt.boxplot([boxdata_mop_2018, boxdata_model_2018, boxdata_obs_2018], positions=[7, 8, 9],
    #                          widths=0.6, showmeans=True,
    #                          patch_artist=True,
    #                          meanline=True,
    #                          labels=['MOPITT', 'MODEL', 'SITE'], whis=[10, 90])
    # plt.title(
    #     'CO Assimilation Results Compared With Observations')
    # colors = ['pink', 'lightblue', 'lightgreen']
    # # for patch, color in zip(bplot['boxes'], colors):
    # #     patch.set_facecolor(color)
    # plt.show()

    # bplot = plt.boxplot([mop_2020, wout_2020, obs_2020], showmeans=True, patch_artist=True, meanline=True,
    #                     labels=['MOPITT', 'MODEL', 'SITE'], whis=[10, 90])
    # plt.title(
    #     'CO Assimilation Results Compared With Observations')
    # plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f ppb'))
    # colors = ['pink', 'lightblue', 'lightgreen']
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)
    #
    # plt.savefig(output_pattern % (str(year), 'box_plot'))
    # plt.show()
    # 最小值(min)，下四分位数(Q1)，中位数(median)，上四分位数(Q3)，最大值(max)
    # y1_smoothed = gaussian_filter1d(y1, sigma=2)
    # plt.plot(x1, y1_smoothed, label=site)
    # y2_smoothed = gaussian_filter1d(y2, sigma=2)
    # plt.plot(x2, y2_smoothed, label="WRF")
    # # y3_smoothed = gaussian_filter1d(y3, sigma=2)
    # # plt.plot(x3, y3_smoothed, label="NO_ASSIM")
    # plt.scatter(x4, y4, label="MOPITT")
    # plt.gcf().autofmt_xdate()
    # plt.title(site)
    # plt.legend()
    # plt.show()
