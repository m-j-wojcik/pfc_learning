from fun_lib import *

with open('config.yml') as file:
    config = yaml.full_load(file)

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time')

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)

    dec_time_res = run_time_resolved_dec_exp2(data, labels, if_save=True,
                                              fname='exp2_dec_timeres',
                                              mode='within_stage')
    dec_time_res_null_ler = run_time_resolved_dec_exp2(data, labels, if_save=True,
                                                       fname='exp2_dec_timeres_iter1', mode='only_ler_null')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
    dec_exp2 = load_data(config['PATHS']['output_path'] + 'exp2_dec_timeres_within_stage.pickle')
    dec_time = dec_exp2['dec_time']

    dec_exp2_null = load_data(config['PATHS']['output_path'] + 'exp2_dec_timeres_iter1_ler_null.pickle')
    dec_time_null = dec_exp2_null['dec_rnd_time']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time')

    dec_time_res = run_time_resolved_sd_exp2(data, labels, if_save=True,
                                              fname='exp2_sd_timeres',
                                              mode='within_stage')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Compute time-averaged shattering dimensionality')
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time')

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)
    sd_avg_stages = []
    for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES']):
        sd_avg_stages.append(shattering_dim_exp2(data_eq[i_stage], labels_eq[i_stage], time_window=[50,100]))
    sd_avg_stages = np.array(sd_avg_stages)

    sd_avg_stages_null_reps = []
    for i_rep in range(config['ANALYSIS_PARAMS']['N_REPS']):
        print('Rep: ', i_rep+1, ' of ', config['ANALYSIS_PARAMS']['N_REPS'])
        x_stage1_and2_rnd, y_stage1_and2_rnd = shuffle_stages(data_eq[0], data_eq[-1], labels_eq[0])
        sd_avg_stages_null = []
        for i_stage in range(2):
            sd_avg_stages_null.append(shattering_dim_exp2(x_stage1_and2_rnd[i_stage], y_stage1_and2_rnd[i_stage], time_window=[50, 100]))
        sd_avg_stages_null_reps.append(sd_avg_stages_null)
    sd_avg_stages_null_reps = np.array(sd_avg_stages_null_reps)

    save_data([sd_avg_stages, sd_avg_stages_null_reps], ['sd_avg_stages', 'sd_avg_stages_null_reps'],
                config['PATHS']['output_path'] + 'exp2_sd_avg_stages_iter1.pickle')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
    dec_exp2 = load_data(config['PATHS']['output_path'] + 'exp2_sd_timeres_within_stage.pickle')
    sd_time = dec_exp2['sd_time']
    sd_avg_stages = load_data(config['PATHS']['output_path'] + 'exp2_sd_avg_stages_iter2.pickle')['sd_avg_stages'].mean(-1)
    sd_avg_stages_null_reps = load_data(config['PATHS']['output_path'] + 'exp2_sd_avg_stages_iter2.pickle')['sd_avg_stages_null_reps'].mean(-1)

    p_val_sd = compute_p_value(sd_avg_stages[0], sd_avg_stages[-1], sd_avg_stages_null_reps[:,0], sd_avg_stages_null_reps[:, -1], tail='greater')


if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Compute firing rate norm')
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time')

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'], n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)
    f_rates_stages = compute_f_rate_norm_stages([data_eq[0], data_eq[-1]], [labels_eq[0], labels_eq[-1]], baseline=False)
    f_rates_stages_null = compute_f_rate_norm_ler_null(data_eq[0], data_eq[-1], labels_eq[0], n_reps=config['ANALYSIS_PARAMS']['N_REPS'], baseline=False)

    save_data([f_rates_stages, f_rates_stages_null], ['f_rates_stages', 'f_rates_stages_null'],
              config['PATHS']['output_path'] + 'exp2_f_rates_stages.pickle')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    print('Load firing rate norm data')
    obj_loaded = load_data(config['PATHS']['output_path'] + 'exp2_f_rates_stages.pickle')
    f_rates_stages = obj_loaded['f_rates_stages']
    f_rates_stages_null = obj_loaded['f_rates_stages_null']


dat = load_data(config['PATHS']['output_path'] + 'exp2_selectivity_dat.pickle')
cos_sim_mat_e = dat['cos_sim_mat_e']
cos_sim_mat = dat['cos_sim_mat']
cos_sim_mat_e_rnd = dat['cos_sim_mat_e_rnd']
cos_sim_mat_rnd = dat['cos_sim_mat_rnd']
epochs_task1_e = dat['epochs_task1_e']
epochs_task2_e = dat['epochs_task2_e']
epochs_task1_l = dat['epochs_task1_l']
epochs_task2_l = dat['epochs_task2_l']
epochs_context1_e = dat['epochs_context1_e']
epochs_context2_e = dat['epochs_context2_e']
epochs_task1_l_irr = dat['epochs_task1_l_irr']
epochs_task2_l_irr = dat['epochs_task2_l_irr']

task_1_e = [np.concatenate([epochs_task1_e[i_stage][:,:1,0], epochs_context1_e[i_stage][:,:1,0]], axis=1) for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
task_2_e = [np.concatenate([epochs_task2_e[i_stage][:,:1,0], epochs_context2_e[i_stage][:,:1,0]], axis=1) for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]

task_1 = [np.concatenate([epochs_task1_l[i_stage][:,:2,0], epochs_task1_l_irr[i_stage][:,:1,0], epochs_task1_l[i_stage][:,2:3,0]], axis=1) for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
task_2 = [np.concatenate([epochs_task2_l[i_stage][:,:2,0], epochs_task2_l_irr[i_stage][:,:1,0], epochs_task2_l[i_stage][:,2:3,0]], axis=1) for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]

tasks = [np.array([task_1[i_stage], task_2[i_stage]]) for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
tasks_e = [np.array([task_1_e[i_stage], task_2_e[i_stage]]) for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]

cos_sim_rnd = compute_lr_null_cos_sim(tasks, reps=10000)
cos_sim_e_rnd = compute_lr_null_cos_sim(tasks_e, reps=10000)


exp1_lis = config['SESSION_NAMES']['sessions_womble_1'][:3] + config['SESSION_NAMES']['sessions_wilfred_1'][:3]
exp2_lis = config['SESSION_NAMES']['sessions_womble_2'][:3] + config['SESSION_NAMES']['sessions_wilfred_2'][:3]
ses_liss = [exp1_lis, exp2_lis]
exp_names = ['exp1', 'exp2']
dat_prop = []
for i_exp in range(2):
    fix_breaks_animals = get_fixation_breaks([ses_liss[i_exp]], experiment_label=exp_names[i_exp])[0]
    no_reward_fix = fix_breaks_animals[:, 2] / fix_breaks_animals[:, 3]
    reward_fix = fix_breaks_animals[:, 1] / fix_breaks_animals[:, 3]
    dat_prop.append(no_reward_fix / reward_fix)


ses_all = combine_session_lists(mode='time', which_exp='exp2')
fix_breaks_animals = get_fixation_breaks([ses_all], experiment_label='exp2')[0]
no_reward_fix = fix_breaks_animals[:, 2] / fix_breaks_animals[:, 3]
reward_fix = fix_breaks_animals[:, 1] / fix_breaks_animals[:, 3]
dat = no_reward_fix / reward_fix
data_prop_s_exp2_set2 = np.array_split(dat, 4)
data_prop_s_exp1_set1 = load_data(config['PATHS']['output_path'] + 'fixation_breaks_prop_s.pickle')['data_prop_s']

# create a grid for figure 3
fig, gs = creat_plot_grid(4, 5, 2)
x_data_time = np.linspace(-0.2, 1.5, dec_time.shape[0])
patch_pars = {'xy': (0.90, 0.0), 'width': 0.1, 'height': 1}
titles = ['context', 'task set', 'shape', 'width', 'xor']
tails_lis_dec = [-1, -1, -1, -1, 1]
tails_lis_xgen = [1, -1, 1, -1, 1]
# plot time-resolved decoding accuracy
for i_panel in range(5):
    ax = line_plot_timevar(gs=gs[0, i_panel], fig=fig, x=x_data_time, y=dec_time[:, i_panel, 0, :].T[None, :, :],
                           color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel] + ' decoding', ylim=[0.45, 0.85],
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars,
                           xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

    diff_obs = dec_time[:, i_panel, 0, -1] - dec_time[:, i_panel, 0, 0]
    diff_null = dec_time_null[:, :, i_panel, 0, -1] - dec_time_null[:, :,  i_panel, 0, 0]
    plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[tails_lis_dec[i_panel]],
                  colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.47,
                  if_smooth=True)

    ax = line_plot_timevar(gs=gs[1, i_panel], fig=fig, x=x_data_time, y=dec_time[:, i_panel, 1, :].T[None, :, :],
                           color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel] + '\ncross-set gen.',
                           ylim=[0.35, 0.75],
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars,
                           xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

    diff_obs = dec_time[:, i_panel, 1, -1] - dec_time[:, i_panel, 1, 0]
    diff_null = dec_time_null[:, :, i_panel, 1, -1] - dec_time_null[:, :,  i_panel, 1, 0]
    plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[tails_lis_xgen[i_panel]],
                  colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.39,
                  if_smooth=True)



ax = line_plot_timevar(gs=gs[3, 1], fig=fig, x=x_data_time, y=sd_time.transpose((2, 0, 1)),
                           color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title='shattering dim.', ylim=[0.48, 0.58],
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars={'xy': (0.0, 0.0), 'width': 0.5, 'height': 1},
                           xlim=[-0.2, 1.5], if_title=True, if_sem=True, xaxese_booo=True)

#add p-value to the plot
ax.text(0.27, 0.8, p_into_stars(p_val_sd), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


# plot the f-rate norm for the different stages
times = np.linspace(-0.5, 2.0, f_rates_stages.shape[-1])
ax = line_plot_timevar(gs=gs[3,2], fig=fig, x=times, y=f_rates_stages[None, :, :], color=['grey', 'black'],
                       xlabel='time (s)', ylabel='f-rate norm', title='firing rate norm', ylim=[None, None],
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=None,
                       patch_pars=None,
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)


diff_obs = f_rates_stages[-1, :] - f_rates_stages[0, :]
diff_null = f_rates_stages_null[:, -1, :] - f_rates_stages_null[:, 0, :]
plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], times, tails_lis=[-1],
              colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=2.5,
              if_smooth=False)
y_lim_r = [-0.5, 0.85]
print('context cos sim')
plot_cos_sim(gs=gs[2,0], fig=fig, obs=cos_sim_mat_e[:, 0], ler_rnd=cos_sim_e_rnd[:, :,  0], rnd_null=cos_sim_mat_e_rnd[:, 0, :], title='selectivity alignment\ncontext', y_lim=y_lim_r, tail='greater')
print('task cos sim')
plot_cos_sim(gs=gs[2,1], fig=fig, obs=cos_sim_mat_e[:, 1], ler_rnd=cos_sim_e_rnd[:, :,  1], rnd_null= cos_sim_mat_e_rnd[:, 1, :], title='selectivity alignment\ntask set', y_lim=y_lim_r, tail='smaller')
print('shape cos sim')
plot_cos_sim(gs=gs[2,2], fig=fig, obs=cos_sim_mat[:, 1], ler_rnd=cos_sim_rnd[:, :, 1], rnd_null=cos_sim_mat_rnd[:, 1, :] , title='selectivity alignment\nshape', y_lim=y_lim_r, tail='greater')
print('width cos sim')
plot_cos_sim(gs=gs[2,3], fig=fig, obs=cos_sim_mat[:, 3], ler_rnd=cos_sim_rnd[:, :, 2], rnd_null= cos_sim_mat_rnd[:, 3, :], title='selectivity alignment\nwidth', y_lim=y_lim_r, tail='smaller')
print('xor cos sim')
plot_cos_sim(gs=gs[2,4], fig=fig, obs=cos_sim_mat[:, 2], ler_rnd=cos_sim_rnd[:, :, 3], rnd_null= cos_sim_mat_rnd[:, 2, :], title='selectivity alignment\nxor', y_lim=y_lim_r, tail='greater')
print('context cos sim')
plot_cos_sim(gs=gs[3,0], fig=fig, obs=cos_sim_mat[:, 0], ler_rnd=cos_sim_rnd[:, :, 0], rnd_null= cos_sim_mat_rnd[:, 0, :], title='selectivity alignment\ncontext', y_lim=y_lim_r, tail='greater')

# plot the proportion of fixation break between rewarded and non-rewarded trials
plot_mean_and_ci_prop(gs=gs[3,3], fig=fig, reward_prop=data_prop_s_exp2_set2, n_perm=10000)


# compute two-sample t-test and pvalue for stage one prop data
plot_data_comparison(gs[3,4], fig, dat_prop[0], dat_prop[1], title='fix. breaks (stage 1)\nexp1 vs exp2', ylim=[None, 7])
fig.tight_layout()

plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_supp_fig_exp2_dec_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.svg')
plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_supp_fig_exp2_dec_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.png', dpi=300)




