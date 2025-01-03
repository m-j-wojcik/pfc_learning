from fun_lib import *
with open('config.yml') as file:
    config = yaml.full_load(file)



print('Load behavioral data')
data_prop_s = load_data(config['PATHS']['output_path'] + 'fixation_breaks_prop_s_' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '.pickle')['data_prop_s']

print('Load temporal decoding data')
data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_timeres_' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '_within_stage.pickle')
shattering_dim_time = data_exp1['shattering_dim_time'].mean(-1).T
decoding_time = data_exp1['decoding_time'].T

shattering_dim_rnd_time = []
decoding_rnd_time = []
decoding_null_time = []
for i_iter in range(1, 3):
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_timeres_' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '_iter' + str(i_iter) + '_ler_null.pickle')
    shattering_dim_rnd_time.append(data_exp1['shattering_dim_rnd_time'])
    decoding_rnd_time.append(data_exp1['decoding_rnd_time'])
    #decoding_null_time.append(data_exp1['decoding_null_time'])

shattering_dim_rnd_time = np.concatenate(shattering_dim_rnd_time, axis=1).mean(-1).T
decoding_rnd_time = np.concatenate(decoding_rnd_time, axis=1).T
#decoding_null_time = np.concatenate(decoding_null_time, axis=-1)

data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_xgen_timeres_' + config['ANALYSIS_PARAMS'][
    'DATA_MODE'] + '_within_stage.pickle')
xgen_decoding_time = data_exp1['xgen_time']

xgen_decoding_null_ler_time = []
for i_iter in range(1, 2):
    data_exp1 = load_data(
        config['PATHS']['output_path'] + 'exp1_xgen_timeres' + config['ANALYSIS_PARAMS']['DATA_MODE'] + '_iter' + str(
            i_iter) + '_ler_null.pickle')
    xgen_decoding_null_ler_time.append(data_exp1['xgen_rnd_time'])

xgen_decoding_null_ler_time = np.concatenate(xgen_decoding_null_ler_time, axis=1).T

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':
    print('Compute cross-width decoding')
    if config['ANALYSIS_PARAMS']['DATA_MODE'] == 'fix_bias':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')
    elif config['ANALYSIS_PARAMS']['DATA_MODE'] == 'stages':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw_stages')

    dec_time_res = run_time_resolved_dec_exp2(data, labels, if_save=True,
                                              fname='exp1_dec_cross_width_timeres',
                                              mode='within_stage', if_exp1=True)

    dec_time_res_null_ler = run_time_resolved_dec_exp2(data, labels, if_save=True,
                                                       fname='exp1_dec_cross_width_timeres_iter2', mode='only_ler_null', if_exp1=True)

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':

    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_dec_cross_width_timeres_within_stage.pickle')
    cross_width_decoding_time = data_exp1['dec_time']

    cross_width_decoding_time_null_ler_time = []
    for i_iter in range(1, 2):
        data_exp1 = load_data(
            config['PATHS']['output_path'] + 'exp1_dec_cross_width_timeres_iter' + str(
                i_iter) + '_ler_null.pickle')
        cross_width_decoding_time_null_ler_time.append(data_exp1['dec_rnd_time'])

    cross_width_decoding_time_null_ler_time = np.concatenate(cross_width_decoding_time_null_ler_time, axis=1).T

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Compute firing rate norm')
    if config['ANALYSIS_PARAMS']['DATA_MODE'] == 'fix_bias':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')
    elif config['ANALYSIS_PARAMS']['DATA_MODE'] == 'stages':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw_stages')


    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'], n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=98)
    f_rates_stages = compute_f_rate_norm_stages([data_eq[0], data_eq[-1]], [labels_eq[0], labels_eq[-1]], baseline=False)
    f_rates_stages_null = compute_f_rate_norm_ler_null(data_eq[0], data_eq[-1], labels_eq[0], n_reps=config['ANALYSIS_PARAMS']['N_REPS'], baseline=False)

    save_data([f_rates_stages, f_rates_stages_null], ['f_rates_stages', 'f_rates_stages_null'],
              config['PATHS']['output_path'] + 'exp1_f_rates_stages.pickle')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    print('Load firing rate norm data')
    obj_loaded = load_data(config['PATHS']['output_path'] + 'exp1_f_rates_stages.pickle')
    f_rates_stages = obj_loaded['f_rates_stages']
    f_rates_stages_null = obj_loaded['f_rates_stages_null']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Run a session-wise decoding analysis')

    variables = [np.array(config['ENCODING_EXP1']['colour']),
                 np.array(config['ENCODING_EXP1']['shape']),
                 np.array(config['ENCODING_EXP1']['width']),
                 np.array(config['ENCODING_EXP1']['xor'])]

    print('Compute firing rate norm')
    if config['ANALYSIS_PARAMS']['DATA_MODE'] == 'fix_bias':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')
    elif config['ANALYSIS_PARAMS']['DATA_MODE'] == 'stages':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw_stages')

    # collapse into one list
    labels = list(chain.from_iterable(labels))
    data = list(chain.from_iterable(data))

    shattering_dim_sessions = np.zeros((len(data), 31))
    decoding_sessions = np.zeros((len(data), 4))
    xgen_sessions = np.zeros((len(data), 4))
    f_rates_sessions = np.zeros((len(data)))
    for i_session in range(len(data)):
        X_session = data[i_session]
        y_session = labels[i_session]
        X_session, y_session = equalise_data_witihn_session([X_session], [y_session], n_splits=1)
        decoding_sess, decoding_dich_sess = get_decoding(data=X_session[0],
                                                         labels=y_session[0],
                                                         variables=variables,
                                                         time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW'],
                                                         method='svm',
                                                         n_jobs=None)


        xgen_sessions[i_session, :] = get_xgen(data=X_session[0],
                                               labels=y_session[0],
                                               variables=variables,
                                               time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW'],
                                               method='svm',
                                               mode='only_rel',
                                               n_jobs=None,
                                               verbose=False
                                               )

        f_rates_ses = compute_f_rate_norm_stages([X_session[0][None, :, :, :]], [y_session[0][None,:]], baseline=True)

        shattering_dim_sessions[i_session, :] = decoding_dich_sess
        decoding_sessions[i_session, :] = decoding_sess[:4]
        f_rates_sessions[i_session] = f_rates_ses[0, config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]:config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]].mean()

    shattering_dim_sessions = shattering_dim_sessions.mean(-1)
    save_data([shattering_dim_sessions, decoding_sessions, xgen_sessions, f_rates_sessions], ['shattering_dim_sessions', 'decoding_sessions', 'xgen_sessions', 'f_rates_sessions'],
              config['PATHS']['output_path'] + 'exp1_decoding_session_wise_'+config['ANALYSIS_PARAMS']['DATA_MODE']+'.pickle')
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    obj_loaded = load_data(config['PATHS']['output_path'] + 'exp1_decoding_session_wise_'+config['ANALYSIS_PARAMS']['DATA_MODE']+'.pickle')
    shattering_dim_sessions = obj_loaded['shattering_dim_sessions']
    xgen_sessions = obj_loaded['xgen_sessions']
    decoding_sessions = obj_loaded['decoding_sessions']
    f_rates_sessions = obj_loaded['f_rates_sessions']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':
    print('Run a time-resolved decoding analysis')
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')

    shattering_dim_time, decoding_time, decoding_null_time = run_time_resolved_decoding(data, labels, if_save=True, fname='exp1_decoding_timeres_fix_bias', mode='within_stage')
    shattering_dim_rnd_time, decoding_rnd_time = run_time_resolved_decoding(data, labels, if_save=True, fname='exp1_decoding_timeres_fix_bias_iter3', mode='only_ler_null', time_resolved=True)

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
    print('Load time-resolved decoding data - sessions sorted by fixation break bias')
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_timeres_fix_bias_within_stage.pickle')
    shattering_dim_time_fix = data_exp1['shattering_dim_time'].mean(-1).T
    decoding_time_fix = data_exp1['decoding_time'].T

    shattering_dim_rnd_time_fix = []
    decoding_rnd_time_fix = []
    for i_iter in range(1, 4):
        data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_timeres_fix_bias_iter' + str(i_iter) + '_ler_null.pickle')
        shattering_dim_rnd_time_fix.append(data_exp1['shattering_dim_rnd_time'])
        decoding_rnd_time_fix.append(data_exp1['decoding_rnd_time'])


    shattering_dim_rnd_time_fix = np.concatenate(shattering_dim_rnd_time_fix, axis=1).mean(-1).T
    decoding_rnd_time_fix = np.concatenate(decoding_rnd_time_fix, axis=1).T

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':

    if config['ANALYSIS_PARAMS']['DATA_MODE'] == 'fix_bias':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')
    elif config['ANALYSIS_PARAMS']['DATA_MODE'] == 'stages':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw_stages')

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=98)
    lsparse_stages = []
    for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES']):
        lsparse = []
        for i_win in range(config['ANALYSIS_PARAMS']['N_WINDOWS']):
            data = data_eq[i_stage][i_win, :, :,
                   config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]:config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]].mean(-1)
            lsparse.append(compute_lifetime_sparseness(data))
        lsparse = np.array(lsparse).mean(0)
        lsparse_stages.append(lsparse)

    save_data([lsparse_stages], ['lsparse_stages'],
              config['PATHS']['output_path'] + 'exp1_sparseness.pickle')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
    obj_loaded = load_data(config['PATHS']['output_path'] + 'exp1_sparseness.pickle')
    lsparse_stages = obj_loaded['lsparse_stages']


if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Computing selectivity coefficients')
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')

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
                                                             cross_val=False,
                                                             )


    save_data([selectivity_coefficients_xval], ['selectivity_coefficients_xval'],
              config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_supp_fig_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) +'.pickle')
    selectivity_coefficients = [selectivity_coefficients_xval[_].mean(-1) for _ in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_supp_fig_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) +'.pickle')
    selectivity_coefficients_xval = data_exp1['selectivity_coefficients_xval']
    selectivity_coefficients = [selectivity_coefficients_xval[_].mean(-1) for _ in range(config['ANALYSIS_PARAMS']['N_STAGES'])]


if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Computing similarity to structured/random models')
    data_exp1 = load_data(config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_supp_fig_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) +'.pickle')
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
              config['PATHS']['output_path'] + 'distance_from_random_exp1_supp_fig_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) +'.pickle')
    p_vals_rnd = p_vals_rnd[0]
    p_vals_str = p_vals_str[0]

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'distance_from_random_exp1_supp_fig_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) +'.pickle')
    rnd_matrix = data_exp1['rnd_matrix']
    p_vals_rnd = data_exp1['p_vals_rnd'][0]
    str_matrix = data_exp1['str_matrix']
    p_vals_str = data_exp1['p_vals_str'][0]

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    if config['ANALYSIS_PARAMS']['DATA_MODE'] == 'fix_bias':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')
    elif config['ANALYSIS_PARAMS']['DATA_MODE'] == 'stages':
        data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw_stages')

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'], n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=98)

    pc_var = run_pca_movewin(data_eq, labels_eq, n_comps=4, factor=[0, 0, 1, 1, 2, 2, 3, 3])
    pc_var_ler_null = run_pca_movewin_null(data_eq[0], data_eq[-1], labels_eq[0], n_comps=4, factor=[0, 0, 1, 1, 2, 2, 3, 3])
    save_data([pc_var, pc_var_ler_null], ['pc_var', 'pc_var_ler_null'], config['PATHS']['output_path'] + 'exp1_supp_fig_pc_var.pickle')
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_supp_fig_pc_var.pickle')
    pc_var = data_exp1['pc_var']
    pc_var_ler_null = data_exp1['pc_var_ler_null']

data_exp1 = load_data(
    config['PATHS']['output_path'] + 'exp1_decoding_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.pickle')
shattering_dim = data_exp1['shattering_dim']
decoding = data_exp1['decoding']
decoding_rnd = data_exp1['decoding_rnd']
shattering_dim_rnd = data_exp1['shattering_dim_rnd']

linear_dichs_idc = np.array([0, 5, 15, 28])
sd_idc = np.arange(shattering_dim.shape[1])
shattering_linear_dims = shattering_dim[:, linear_dichs_idc]
shattering_linear_dims = np.concatenate([shattering_linear_dims, decoding[:, :3]], axis=1)
shattering_linear_dims_rnd = np.concatenate([decoding_rnd[:, :, :3], shattering_dim_rnd[:, :, linear_dichs_idc]], axis=2)
mask = np.ones(sd_idc.shape, dtype=bool)
mask[linear_dichs_idc] = False
shattering_nonlinear_dims = shattering_dim[:, mask]
shattering_nonlinear_dims_rnd = shattering_dim_rnd[:, :, mask]


if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')

    xgen_decoding = run_time_resolved_xgen(data, labels, if_save=True,
                                           fname='exp1_xgen_timeres_fix_bias',
                                           mode='within_stage_without_null')
    xgen_decoding_null_ler = run_time_resolved_xgen(data, labels, if_save=True,
                                                    fname='exp1_xgen_timeres_fix_bias_iter2', mode='only_ler_null', time_resolved=True)

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
    print('Load time-resolved decoding data')
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_xgen_timeres_fix_bias_within_stage.pickle')
    xgen_decoding = data_exp1['xgen_time']

    xgen_decoding_null_ler = []
    for i_iter in range(1, 3):
        data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_xgen_timeres_fix_bias_iter' + str(i_iter) + '_ler_null.pickle')
        xgen_decoding_null_ler.append(data_exp1['xgen_rnd_time'])

    xgen_decoding_null_ler = np.concatenate(xgen_decoding_null_ler, axis=1).T

# create a grid for the decoding supp. figure
fig, gs = creat_plot_grid(4, 5, 2)

times = np.linspace(-0.5, 2.0, f_rates_stages.shape[-1])

# prepare decoding data for next three panels
x_data_time = np.linspace(-0.2, 1.5, shattering_dim_time.shape[-1])
patch_pars = {'xy': (0.90, 0.0), 'width': 0.1, 'height': 1}
titles = ['colour decoding', 'shape decoding', 'width decoding', 'xor decoding', 'shattering dimensionality']
y_lims = [0.45, 0.95]
tails_lis = [-1, -1, -1, 1]

ax = line_plot_timevar(gs=gs[0,3], fig=fig, x=x_data_time, y=cross_width_decoding_time[:, -1, :].T[None, :, :], color=['grey', 'black'],
                       xlabel='time (s)', ylabel='accuracy', title='xor cross-width\ngen. decoding', ylim=y_lims,
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                       patch_pars=patch_pars,
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

diff_obs = cross_width_decoding_time[:, -1, -1] - cross_width_decoding_time[:, -1, 0]
diff_null = cross_width_decoding_time_null_ler_time[-1,-1,:,:] - cross_width_decoding_time_null_ler_time[0,-1,:,:]
diff_null = diff_null.T
plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[1],
              colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.47,
              if_smooth=False, variable_name='xor cross-width\ngen. decoding')


ax = line_plot_timevar(gs=gs[0,2], fig=fig, x=x_data_time, y=decoding_time[2, :, :][None, :, :], color=['grey', 'black'],
                       xlabel='time (s)', ylabel='accuracy', title=titles[2], ylim=y_lims,
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                       patch_pars=patch_pars,
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

diff_obs = decoding_time[2, -1, :] - decoding_time[2, 0, :]
diff_null = decoding_rnd_time[2, -1, :, :] - decoding_rnd_time[2, 0, :, :]
plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[tails_lis[2]],
              colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.47,
              if_smooth=False, variable_name=titles[2])

ax = line_plot_timevar(gs=gs[0, 4], fig=fig, x=x_data_time,
                       y=xgen_decoding_time[:, :, 2].T[None, :, :], color=['grey', 'black'],
                       xlabel='time (s)', ylabel='accuracy', title='width\ncross. gen. dec.',
                       ylim=y_lims,
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                       patch_pars=patch_pars,
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

diff_obs = xgen_decoding_time[:, -1, 2] - xgen_decoding_time[:, 0, 2]
diff_null = xgen_decoding_null_ler_time[2, -1, :, :] - xgen_decoding_null_ler_time[2, 0, :,:]
plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[tails_lis[2]],
              colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.47,
              if_smooth=False, variable_name='width\ncross. gen. dec.')

# plote the time-resolved decoding for the fixation bias split of data
for i_panel in range(4):
    ax = line_plot_timevar(gs=gs[2, i_panel], fig=fig, x=x_data_time, y=decoding_time_fix[i_panel, :, :][None, :, :], color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel], ylim=y_lims,
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars,
                           xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

    diff_obs = decoding_time_fix[i_panel, -1, :] - decoding_time_fix[i_panel, 0, :]
    diff_null = decoding_rnd_time_fix[i_panel, -1, :, :] - decoding_rnd_time_fix[i_panel, 0, :, :]
    plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[tails_lis[i_panel]],
                  colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.47,
                  if_smooth=False)

# plot the shattering dimensionality
ax = line_plot_timevar(gs=gs[2, 4], fig=fig, x=x_data_time, y=shattering_dim_time_fix[None, :, :], color=['grey', 'black'],
                          xlabel='time (s)', ylabel='accuracy', title='shattering dimensionality', ylim=y_lims,
                            xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                            patch_pars=patch_pars,
                            xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

diff_obs = shattering_dim_time_fix[-1, :] - shattering_dim_time_fix[0, :]
diff_null = shattering_dim_rnd_time_fix[-1, :, :] - shattering_dim_rnd_time_fix[ 0, :, :]
plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[-1],
              colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.47,
              if_smooth=False)


# plot the f-rate norm for the different stages
ax = line_plot_timevar(gs=gs[1,-1], fig=fig, x=times, y=f_rates_stages[None, :, :], color=['grey', 'black'],
                       xlabel='time (s)', ylabel='norm\n(standardised)', title='firing rate norm', ylim=[4.7, None],
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=None,
                       patch_pars={'xy': (0.90, -3), 'width': 0.1, 'height': 20},
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

diff_obs = f_rates_stages[-1, :] - f_rates_stages[0, :]
diff_null = f_rates_stages_null[:, -1, :] - f_rates_stages_null[:, 0, :]
plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], times, tails_lis=[-1],
                  colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=5,
              if_smooth=False)

print('Sparseness analysis')
plot_sparseness(gs[1, -3], fig, lsparse_stages[0], lsparse_stages[-1])

# ses_all = combine_session_lists(mode='fix_bias', which_exp='exp1')
# fix_breaks_animals = get_fixation_breaks([ses_all], experiment_label='exp1')[0]
# no_reward_fix = fix_breaks_animals[:, 2] / fix_breaks_animals[:, 3]
# reward_fix = fix_breaks_animals[:, 1] / fix_breaks_animals[:, 3]
# dat = no_reward_fix / reward_fix
# data_prop_s = np.array_split(dat, 4)
# mean_fix_prop = [str(round(np.nanmean(_), 1)) for _ in data_prop_s]
mean_fix_prop = ['1.1', '1.3', '1.9', '3.3']
plot_distance(gs[3, 0], fig, rnd_matrix, 'distance from\nrandom', pvals=p_vals_rnd, x_stages_ticks=mean_fix_prop, x_label='trial termination bias')
plot_distance(gs[3, 1], fig, str_matrix, 'distance from\nstructured', pvals=p_vals_str , x_stages_ticks=mean_fix_prop, x_label='trial termination bias')

print('PCA analysis')
plot_pca(gs[1, 0], fig, pc_var, pc_var_ler_null, title='First PC', y_lim=[.40,.60])


x_data = list(chain.from_iterable(data_prop_s))

plot_regression_sup(x_data, xgen_sessions[:,-1], gs[1, 3], fig, xlabel='fixation break bias', ylabel='xor decoding', title='session-wise corr.')

print('Linear vs nonlinear dimensionality')
plot_lin_nonlin_dims(gs[1, 1], fig, shattering_linear_dims, shattering_nonlinear_dims, shattering_linear_dims_rnd, shattering_nonlinear_dims_rnd, y_lim=[0.55, 0.75], title='dimensionality\n(linear vs nonlin.)')

titles = ['colour', 'shape', 'width', 'xor']
# plot time-resolved xgen decoding accuracy
for i_panel in range(3):
    skip_width = 0
    if i_panel == 2:
        skip_width = 1
    ax = line_plot_timevar(gs=gs[3, i_panel+2], fig=fig, x=x_data_time, y=xgen_decoding[:, :, i_panel+skip_width].T[None, :, :], color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel+skip_width] + '\ncross. gen. dec.', ylim=[0.3, 0.95],
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars,
                           xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)

    diff_obs = xgen_decoding[:, -1, i_panel+skip_width] - xgen_decoding[:, 0, i_panel+skip_width]
    diff_null = xgen_decoding_null_ler[i_panel+skip_width, -1, :, :] - xgen_decoding_null_ler[i_panel+skip_width, 0, :, :]
    plot_sig_bars(ax, diff_obs[None, :], diff_null[None, :, :], x_data_time, tails_lis=[tails_lis[i_panel+skip_width]],
                  colour_lis=['black'], p_threshold=0.05, plot_chance_lvl=0.35,
                  if_smooth=False)
fig.tight_layout()


plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_supp_fig_exp1_dec' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.svg')
plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_supp_fig_exp1_dec' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.png', dpi=300)







