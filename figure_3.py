from fun_lib import *
with open('config.yml') as file:
    config = yaml.full_load(file)

# compute the proportion of fixation breaks between rewarded and non-rewarded trials as a function of learning
if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Running behavioural analysis')
    ses_all = combine_session_lists(mode=config['ANALYSIS_PARAMS']['DATA_MODE'], which_exp='exp1', combine_all=False)
    fix_breaks_animals = get_fixation_breaks(ses_all, experiment_label='exp1')
    data_prop_s = []
    for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES']):
        no_reward_fix = fix_breaks_animals[i_stage][:, 2] / fix_breaks_animals[i_stage][:, 3]
        reward_fix = fix_breaks_animals[i_stage][:, 1] / fix_breaks_animals[i_stage][:, 3]
        dat = no_reward_fix / reward_fix
        data_prop_s.append(dat)
    save_data([data_prop_s], ['data_prop_s'], config['PATHS']['output_path'] + 'fixation_breaks_prop_s_' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_prop_s = load_data(config['PATHS']['output_path'] + 'fixation_breaks_prop_s_' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')['data_prop_s']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':
    print('Run a time-resolved decoding analysis')
    if config['ANALYSIS_PARAMS']['DATA_MODE'] == 'fix_bias':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')
    elif config['ANALYSIS_PARAMS']['DATA_MODE'] == 'stages':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw_stages')

    shattering_dim_time, decoding_time, decoding_null_time = run_time_resolved_decoding(data, labels, if_save=True, fname='exp1_decoding_timeres_' + config['ANALYSIS_PARAMS']['DATA_MODE'], mode='within_stage')
    shattering_dim_rnd_time, decoding_rnd_time = run_time_resolved_decoding(data, labels, if_save=True, fname='exp1_decoding_timeres_' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '_iter3', mode='only_ler_null', time_resolved=True)

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
    print('Load time-resolved decoding data')
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_timeres_' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '_within_stage.pickle')
    shattering_dim_time = data_exp1['shattering_dim_time'].mean(-1).T
    decoding_time = data_exp1['decoding_time'].T

    shattering_dim_rnd_time = []
    decoding_rnd_time = []
    for i_iter in range(1, 4):
        data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_timeres_' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '_iter' + str(i_iter) + '_ler_null.pickle')
        shattering_dim_rnd_time.append(data_exp1['shattering_dim_rnd_time'])
        decoding_rnd_time.append(data_exp1['decoding_rnd_time'])


    shattering_dim_rnd_time = np.concatenate(shattering_dim_rnd_time, axis=1).mean(-1).T
    decoding_rnd_time = np.concatenate(decoding_rnd_time, axis=1).T

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Computing selectivity coefficients')
    if config['ANALYSIS_PARAMS']['DATA_MODE'] == 'fix_bias':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')
    elif config['ANALYSIS_PARAMS']['DATA_MODE'] == 'stages':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw_stages')




    var_encoding_sel = [config['SEL_ENCODING_EXP1']['colour_sel'],
                        config['SEL_ENCODING_EXP1']['shape_sel'],
                        config['SEL_ENCODING_EXP1']['xor_sel'],
                        config['SEL_ENCODING_EXP1']['width_sel'],
                        config['SEL_ENCODING_EXP1']['int_irrel_sel']]


    selectivity_coefficients_xval, _ = get_betas_cross_val_2(data=data,
                                                             labels=labels,
                                                             condition_labels=var_encoding_sel,
                                                             normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                                             time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW'],
                                                             task=config['ANALYSIS_PARAMS']['WHICH_TASK_SEL'],
                                                             )


    save_data([selectivity_coefficients_xval], ['selectivity_coefficients_xval'],
              config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1])  + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')
    selectivity_coefficients = [selectivity_coefficients_xval[_].mean(-1) for _ in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')
    selectivity_coefficients_xval = data_exp1['selectivity_coefficients_xval']
    selectivity_coefficients = [selectivity_coefficients_xval[_].mean(-1) for _ in range(config['ANALYSIS_PARAMS']['N_STAGES'])]


if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Computing similarity to structured/random models')
    data_exp1 = load_data(config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')
    selectivity_coefficients_xval = data_exp1['selectivity_coefficients_xval']
    rnd_matrix, p_vals_rnd = compute_dist_random(selectivity_coeffs_stages=selectivity_coefficients_xval,
                                                 n_bootstraps=config['ANALYSIS_PARAMS']['N_BOOTSTRAP_SEL'],
                                                 rnd_model=config['ANALYSIS_PARAMS']['RND_MODEL_SEL'],
                                                 metric=config['ANALYSIS_PARAMS']['METRIC_SEL'],
                                                 design_model=config['ANALYSIS_PARAMS']['DESIGN_ENCODING_SEL'],
                                                 bon_correction=True,
                                                 )

    str_matrix, p_vals_str = compute_dist_structured(selectivity_coeffs_stages=selectivity_coefficients_xval,
                                                     n_bootstraps=config['ANALYSIS_PARAMS']['N_BOOTSTRAP_SEL'],
                                                     rnd_model=config['ANALYSIS_PARAMS']['RND_MODEL_SEL'],
                                                     metric=config['ANALYSIS_PARAMS']['METRIC_SEL'],
                                                     design_model=config['ANALYSIS_PARAMS']['DESIGN_ENCODING_SEL'],
                                                     bon_correction=True,
                                                     )


    save_data([rnd_matrix, p_vals_rnd, str_matrix, p_vals_str],
              ['rnd_matrix', 'p_vals_rnd', 'str_matrix', 'p_vals_str'],
              config['PATHS']['output_path'] + 'distance_from_random_exp1_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')
    p_vals_rnd = p_vals_rnd[0]
    p_vals_str = p_vals_str[0]

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'distance_from_random_exp1_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')
    rnd_matrix = data_exp1['rnd_matrix']
    p_vals_rnd = data_exp1['p_vals_rnd'][0]
    str_matrix = data_exp1['str_matrix']
    p_vals_str = data_exp1['p_vals_str'][0]

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Running the time-averaged decoding analysis')

    if config['ANALYSIS_PARAMS']['DATA_MODE'] == 'fix_bias':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')
    elif config['ANALYSIS_PARAMS']['DATA_MODE'] == 'stages':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw_stages')


    variables = [np.array(config['ENCODING_EXP1']['colour']),
                 np.array(config['ENCODING_EXP1']['shape']),
                 np.array(config['ENCODING_EXP1']['width']),
                 np.array(config['ENCODING_EXP1']['xor'])]
    print('Warning, correcting labels for cross-gen')
    labels = correct_labels_for_cross_gen(labels)

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'], n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=98)
    shattering_dim, decoding, xgen_decoding, decoding_null, xgen_null = run_moving_window_decoding(data_eq, labels_eq, variables,
                                                                                        time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW'],
                                                                                        method='svm', n_reps=config['ANALYSIS_PARAMS']['N_REPS'],
                                                                                        if_null=True)
    shattering_dim_rnd, decoding_rnd, xgen_decoding_rnd = run_moving_window_decoding_ler_null(data_eq[0], data_eq[-1],
                                                                                              labels_eq[0], variables,
                                                                                              time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW'],
                                                                                              n_reps=config['ANALYSIS_PARAMS']['N_REPS'])
    data_exp1 = [shattering_dim, decoding, xgen_decoding, decoding_null, xgen_null, shattering_dim_rnd, decoding_rnd,
                 xgen_decoding_rnd]
    names_exp1 = ['shattering_dim', 'decoding', 'xgen_decoding', 'decoding_null', 'xgen_null', 'shattering_dim_rnd', 'decoding_rnd',
                  'xgen_decoding_rnd']
    save_data(data_exp1, names_exp1,
              config['PATHS']['output_path'] + 'exp1_decoding_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')
    shattering_dim = data_exp1['shattering_dim']
    decoding = data_exp1['decoding']
    decoding_null = data_exp1['decoding_null']
    xgen_null_avg = data_exp1['xgen_null']
    xgen_decoding_avg = data_exp1['xgen_decoding']
    shattering_dim_rnd = data_exp1['shattering_dim_rnd']
    decoding_rnd = data_exp1['decoding_rnd']
    xgen_decoding_rnd_avg = data_exp1['xgen_decoding_rnd']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':

    if config['ANALYSIS_PARAMS']['DATA_MODE'] == 'fix_bias':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')
    elif config['ANALYSIS_PARAMS']['DATA_MODE'] == 'stages':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw_stages')

    xgen_decoding = run_time_resolved_xgen(data, labels, if_save=True, fname='exp1_xgen_timeres_' + config['ANALYSIS_PARAMS']['DATA_MODE'], mode='within_stage_without_null')
    xgen_decoding_null_ler = run_time_resolved_xgen(data, labels, if_save=True, fname='exp1_xgen_timeres' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '_iter1', mode='only_ler_null')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
    print('Load time-resolved decoding data')
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_xgen_timeres_'+ config['ANALYSIS_PARAMS']['DATA_MODE'] + '_within_stage.pickle')
    xgen_decoding = data_exp1['xgen_time']

    xgen_decoding_null_ler = []
    for i_iter in range(1, 2):
        data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_xgen_timeres' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '_iter' + str(i_iter) + '_ler_null.pickle')
        xgen_decoding_null_ler.append(data_exp1['xgen_rnd_time'])

    xgen_decoding_null_ler = np.concatenate(xgen_decoding_null_ler, axis=1).T

# create a grid for figure 3
fig, gs = creat_plot_grid(5, 4, 2)

# plot the proportion of fixation break between rewarded and non-rewarded trials
plot_mean_and_ci_prop(gs[0], fig, data_prop_s, n_perm=10000)

# prepare decoding data for next three panels
x_data_time = np.linspace(-0.2, 1.5, shattering_dim_time.shape[-1])
patch_pars = {'xy': (0.90, 0.0), 'width': 0.1, 'height': 1}
titles = ['colour decoding', 'shape decoding', 'width decoding', 'xor decoding', 'shattering dimensionality']
y_lims = [0.45, 0.95]
tails_lis = [-1, -1, -1, 1]

# plot time-resolved decoding accuracy
for i_panel in range(3):
    skip_width = 0
    if i_panel == 2:
        skip_width = 1
    ax = line_plot_timevar(gs=gs[i_panel+1], fig=fig, x=x_data_time, y=decoding_time[i_panel+skip_width, :, :][None, :, :], color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel+skip_width], ylim=y_lims,
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars,
                           xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

    diff_obs = decoding_time[i_panel+skip_width, -1, :] - decoding_time[i_panel+skip_width, 0, :]
    diff_null = decoding_rnd_time[i_panel+skip_width, -1, :, :] - decoding_rnd_time[i_panel+skip_width, 0, :, :]
    plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[tails_lis[i_panel+skip_width]],
                  colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.47,
                  if_smooth=False, variable_name=titles[i_panel+skip_width])

titles = ['colour', 'shape', 'width', 'xor']
# plot time-resolved xgen decoding accuracy
for i_panel in range(3):
    skip_width = 0
    if i_panel == 2:
        skip_width = 1
    ax = line_plot_timevar(gs=gs[1, i_panel+1], fig=fig, x=x_data_time, y=xgen_decoding[:, :, i_panel+skip_width].T[None, :, :], color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel+skip_width] + '\ncross. gen. dec.', ylim=[0.3, 0.95],
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars,
                           xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

    diff_obs = xgen_decoding[:, -1, i_panel+skip_width] - xgen_decoding[:, 0, i_panel+skip_width]
    diff_null = xgen_decoding_null_ler[i_panel+skip_width, -1, :, :] - xgen_decoding_null_ler[i_panel+skip_width, 0, :, :]
    plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[tails_lis[i_panel+skip_width]],
                  colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.35,
                  if_smooth=False, variable_name= 'cross. gen. dec.'+ titles[i_panel+skip_width])


# plot the shattering dimensionality
ax = line_plot_timevar(gs=gs[1, 0], fig=fig, x=x_data_time, y=shattering_dim_time[None, :, :], color=['grey', 'black'],
                          xlabel='time (s)', ylabel='accuracy', title='neural\ndimensionality', ylim=[0.45, 0.75],
                            xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                            patch_pars=patch_pars,
                            xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

diff_obs = shattering_dim_time[-1, :] - shattering_dim_time[0, :]
diff_null = shattering_dim_rnd_time[-1, :, :] - shattering_dim_rnd_time[ 0, :, :]
plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[-1],
              colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.47,
              if_smooth=False, variable_name= "shattering dimensionality")


x_data_r2 = [selectivity_coefficients[0][:, config['ANALYSIS_PARAMS']['COLOUR_ID']], selectivity_coefficients[0][:, config['ANALYSIS_PARAMS']['COLOUR_ID']],
             selectivity_coefficients[-1][:, config['ANALYSIS_PARAMS']['COLOUR_ID']], selectivity_coefficients[-1][:, config['ANALYSIS_PARAMS']['COLOUR_ID']]]
y_data_r2 = [selectivity_coefficients[0][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], selectivity_coefficients[0][:, -1],
             selectivity_coefficients[-1][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], selectivity_coefficients[-1][:, -1]]
x_data_r3 = [selectivity_coefficients[0][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], None, selectivity_coefficients[-1][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], None]
y_data_r3 = [selectivity_coefficients[0][:, -1], None, selectivity_coefficients[-1][:, -1], None]

overlay_models_r2 = ['random', 'random', 'minimal1', 'minimal2']
xaxis_label_lis_r2 = ['colour selectivity', 'colour selectivity', 'colour selectivity', 'colour selectivity']
yaxis_label_lis_r2 = ['shape selectivity', 'xor selectivity', 'shape selectivity', 'xor selectivity']
xaxis_label_lis_r3 = ['shape selectivity', None, 'shape selectivity', None]
yaxis_label_lis_r3 = ['xor selectivity', None, 'xor selectivity', None]
overlay_models_r3 = ['random', None, 'minimal2', None]
cov_dat = [None, selectivity_coefficients[0], None, selectivity_coefficients[-1]]

for i_panel in range(4):
    plot_scatter(gs=gs[2, i_panel], fig=fig, x=x_data_r2[i_panel], y=y_data_r2[i_panel], scale=.25,
                 xaxis_label=xaxis_label_lis_r2[i_panel], yaxis_label=yaxis_label_lis_r2[i_panel],
                 overlay_model=overlay_models_r2[i_panel])
    # if i_panel is odd
    if i_panel % 2 == 1:
        plot_cov(gs=gs[3, i_panel], fig=fig, dat=cov_dat[i_panel], scale=0.007)

    if i_panel % 2 == 0:
        plot_scatter(gs=gs[3, i_panel], fig=fig, x=x_data_r3[i_panel], y=y_data_r3[i_panel], scale=.25,
                     xaxis_label=xaxis_label_lis_r3[i_panel], yaxis_label=yaxis_label_lis_r3[i_panel],
                     overlay_model=overlay_models_r3[i_panel])

plot_distance(gs[4, :2], fig, rnd_matrix, 'distance from random', pvals=p_vals_rnd)
plot_distance(gs[4, 2:], fig, str_matrix, 'distance from structured', pvals=p_vals_str)

plt.tight_layout()
# save the figure

plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_figure_3_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.svg')
plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_figure_3_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.png', dpi=300)
