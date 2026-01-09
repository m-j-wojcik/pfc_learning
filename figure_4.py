from fun_lib import *
import matplotlib
matplotlib.use('TkAgg')
with open('config.yml') as file:
    config = yaml.full_load(file)
#%%

os.makedirs(config['PATHS']['out_template_figures'], exist_ok=True)
os.makedirs(config['PATHS']['output_path'], exist_ok=True)

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_PREPROCESSING'] == 'run':
    if config['ANALYSIS_PARAMS']['N_STAGES'] < 5:
        ses_all = combine_session_lists(mode='stages', which_exp='exp2', combine_all=False)
    else:
        # when discretising learning into more than 4 stages use a sliding window approach to maintain sufficient statistical
        # power (have at least ~70 neurons per stage)
        ses_all = combine_session_lists(mode='sliding_window', which_exp='exp2', combine_all=False)

    data, labels = get_data_stages(observe_or_run='run', session_list=ses_all,
                                   file_name='exp2_f_rates_ses_time_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages', return_data=True)

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages')

    taskset = np.array(config['ENCODING_EXP2']['taskset'])
    context = np.array(config['ENCODING_EXP2']['context'])
    shape = np.array(config['ENCODING_EXP2']['shape'])
    width = np.array(config['ENCODING_EXP2']['width'])
    xor = np.array(config['ENCODING_EXP2']['xor'])
    xor2 = np.array(config['ENCODING_EXP2']['xor2'])

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)
    print('Running decoding analysis...')
    scores_context_early, scores_set_early = run_decoding_colour_locked(data_eq, labels_eq,
                                                                        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'],
                                                                        factors=[context, taskset])

    print('Constructing null distribution for decoding analysis...')
    scores_context_rnd_early, scores_set_rnd_early = run_decoding_ler_null_colour_locked(data_eq, labels_eq,
                                                                                         time_window=
                                                                                         config['ANALYSIS_PARAMS'][
                                                                                             'TIME_WINDOW_EXP2_COL_LOCK'],
                                                                                         n_reps=
                                                                                         config['ANALYSIS_PARAMS'][
                                                                                             'N_REPS'],
                                                                                         factors=[context, taskset])

    dec_data_exp2_early = [scores_context_early, scores_set_early, scores_context_rnd_early, scores_set_rnd_early]
    names_early = ['scores_context_early', 'scores_set_early', 'scores_context_rnd_early', 'scores_set_rnd_early']
    save_data(dec_data_exp2_early, names_early,
              config['PATHS']['output_path'] + 'exp2_decoding_collocked_' + str(
                  config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' + str(
                  config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) + '_stages_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + '.pickle')

    scores_context_late, scores_set_late, scores_shape_late, scores_width_late, scores_xor_late, scores_xor2_late = run_decoding_shape_locked(
        data_eq,
        labels_eq,
        config[
            'ANALYSIS_PARAMS'][
            'TIME_WINDOW_EXP2_SHAPE_LOCK'],
        factors=[context,
                 taskset,
                 shape,
                 width,
                 xor, xor2])

    scores_context_rnd_late, scores_set_rnd_late, scores_shape_rnd_late, scores_width_rnd_late, scores_xor_rnd_late, scores_xor2_rnd_late = run_decoding_ler_null_shape_locked(
        data_eq, labels_eq, time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'],
        n_reps=config['ANALYSIS_PARAMS']['N_REPS'],
        factors=[context, taskset, shape, width, xor, xor2])


    dec_data_exp2_late = [scores_context_late, scores_set_late, scores_shape_late, scores_width_late, scores_xor_late, scores_xor2_late,
                          scores_context_rnd_late, scores_set_rnd_late, scores_shape_rnd_late, scores_width_rnd_late,
                          scores_xor_rnd_late, scores_xor2_rnd_late]
    names_late = ['scores_context_late', 'scores_set_late', 'scores_shape_late', 'scores_width_late', 'scores_xor_late', 'scores_xor2_late',
                  'scores_context_rnd_late', 'scores_set_rnd_late', 'scores_shape_rnd_late', 'scores_width_rnd_late',
                  'scores_xor_rnd_late', 'scores_xor2_rnd_late']
    save_data(dec_data_exp2_late, names_late,
              config['PATHS']['output_path'] + 'exp2_decoding_shapelocked_' + str(
                  config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' + str(
                  config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + '.pickle')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    dat_col_locked = load_data(config['PATHS']['output_path'] + 'exp2_decoding_collocked_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) + '_stages_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + '.pickle')
    dat_shape_locked = load_data(config['PATHS']['output_path'] + 'exp2_decoding_shapelocked_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + '.pickle')

    scores_context_early = dat_col_locked['scores_context_early']
    scores_set_early = dat_col_locked['scores_set_early']
    scores_context_rnd_early = dat_col_locked['scores_context_rnd_early']
    scores_set_rnd_early = dat_col_locked['scores_set_rnd_early']

    scores_context_late = dat_shape_locked['scores_context_late']
    scores_set_late = dat_shape_locked['scores_set_late']
    scores_shape_late = dat_shape_locked['scores_shape_late']
    scores_width_late = dat_shape_locked['scores_width_late']
    scores_xor_late = dat_shape_locked['scores_xor_late']
    scores_context_rnd_late = dat_shape_locked['scores_context_rnd_late']
    scores_set_rnd_late = dat_shape_locked['scores_set_rnd_late']
    scores_shape_rnd_late = dat_shape_locked['scores_shape_rnd_late']
    scores_width_rnd_late = dat_shape_locked['scores_width_rnd_late']
    scores_xor_rnd_late = dat_shape_locked['scores_xor_rnd_late']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    var_encoding_sel = [config['SEL_ENCODING_EXP1']['colour_sel'],
                        config['SEL_ENCODING_EXP1']['shape_sel'],
                        config['SEL_ENCODING_EXP1']['xor_sel'],
                        config['SEL_ENCODING_EXP1']['width_sel'],
                        config['SEL_ENCODING_EXP1']['int_irrel_sel']]

    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages')

    epochs_task1_l, epochs_task1_l_irr = get_betas_cross_val_2(data=data,
                                                               labels=labels,
                                                               condition_labels=var_encoding_sel,
                                                               normalisation=config['ANALYSIS_PARAMS'][
                                                                   'NORMALISATION_SEL'],
                                                               time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'],
                                                               task='task_1',
                                                               )

    epochs_task2_l, epochs_task2_l_irr = get_betas_cross_val_2(data=data,
                                                               labels=labels,
                                                               condition_labels=var_encoding_sel,
                                                               normalisation=config['ANALYSIS_PARAMS'][
                                                                   'NORMALISATION_SEL'],
                                                               time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'],
                                                               task='task_2',
                                                               )

    epochs_task1_e, _ = get_betas_cross_val_2(data=data,
                                              labels=labels,
                                              condition_labels=var_encoding_sel,
                                              normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                              time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'],
                                              task='task_1',
                                              )

    epochs_task2_e, _ = get_betas_cross_val_2(data=data,
                                              labels=labels,
                                              condition_labels=var_encoding_sel,
                                              normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                              time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'],
                                              task='task_2',
                                              )

    epochs_context1_e, _ = get_betas_cross_val_2(data=data,
                                                 labels=labels,
                                                 condition_labels=var_encoding_sel,
                                                 normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                                 time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'],
                                                 task='context_1',
                                                 )

    epochs_context2_e, _ = get_betas_cross_val_2(data=data,
                                                 labels=labels,
                                                 condition_labels=var_encoding_sel,
                                                 normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                                 time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'],
                                                 task='context_2',
                                                 )

    cos_sim_mat = np.zeros((config['ANALYSIS_PARAMS']['N_STAGES'], 4))
    cos_sim_mat_rnd = np.zeros((config['ANALYSIS_PARAMS']['N_STAGES'], 4, config['ANALYSIS_PARAMS']['N_REPS']))

    cos_sim_mat_e = np.zeros((config['ANALYSIS_PARAMS']['N_STAGES'], 2))
    cos_sim_mat_e_rnd = np.zeros((config['ANALYSIS_PARAMS']['N_STAGES'], 2, config['ANALYSIS_PARAMS']['N_REPS']))

    for p in range(config['ANALYSIS_PARAMS']['N_STAGES']):
        colour_task_1 = epochs_task1_l[p][:, 0, 0]
        colour_task_2 = epochs_task2_l[p][:, 0, 0]

        shape_task_1 = epochs_task1_l[p][:, 1, 0]
        shape_task_2 = epochs_task2_l[p][:, 1, 0]

        width_task_1 = epochs_task1_l_irr[p][:, 1, 0]
        width_task_2 = epochs_task2_l_irr[p][:, 1, 0]

        xor_task_1 = epochs_task1_l[p][:, 2, 0]
        xor_task_2 = epochs_task2_l[p][:, 2, 0]

        cos_sim_mat[p, 0] = cos_sim(colour_task_1, colour_task_2)
        cos_sim_mat[p, 1] = cos_sim(shape_task_1, shape_task_2)
        cos_sim_mat[p, 2] = cos_sim(xor_task_1, xor_task_2)
        cos_sim_mat[p, 3] = cos_sim(width_task_1, width_task_2)

        colour_task_1_e = epochs_task1_e[p][:, 0, 0]
        colour_task_2_e = epochs_task2_e[p][:, 0, 0]
        cos_sim_mat_e[p, 0] = cos_sim(colour_task_1_e, colour_task_2_e)

        set_context_1_e = epochs_context1_e[p][:, 0, 0]
        set_context_2_e = epochs_context2_e[p][:, 0, 0]
        cos_sim_mat_e[p, 1] = cos_sim(set_context_1_e, set_context_2_e)

        for rep in range(config['ANALYSIS_PARAMS']['N_REPS']):
            colour_task_2_rnd = epochs_task2_l[p][:, 0, 0].copy()
            shape_task_2_rnd = epochs_task2_l[p][:, 1, 0].copy()
            width_task_2_rnd = epochs_task2_l_irr[p][:, 1, 0].copy()
            xor_task_2_rnd = epochs_task2_l[p][:, 2, 0].copy()
            colour_task_2_e_rnd = epochs_task2_e[p][:, 0, 0].copy()
            set_context_2_e_rnd = epochs_context2_e[p][:, 0, 0].copy()

            random.shuffle(colour_task_2_rnd)
            random.shuffle(shape_task_2_rnd)
            random.shuffle(width_task_2_rnd)
            random.shuffle(xor_task_2_rnd)
            random.shuffle(colour_task_2_e_rnd)
            random.shuffle(set_context_2_e_rnd)

            cos_sim_mat_rnd[p, 0, rep] = cos_sim(colour_task_1, colour_task_2_rnd)
            cos_sim_mat_rnd[p, 1, rep] = cos_sim(shape_task_1, shape_task_2_rnd)
            cos_sim_mat_rnd[p, 2, rep] = cos_sim(xor_task_1, xor_task_2_rnd)
            cos_sim_mat_rnd[p, 3, rep] = cos_sim(width_task_1, width_task_2_rnd)
            cos_sim_mat_e_rnd[p, 0, rep] = cos_sim(colour_task_1_e, colour_task_2_e_rnd)
            cos_sim_mat_e_rnd[p, 1, rep] = cos_sim(set_context_1_e, set_context_2_e_rnd)

    p_vals_l = np.ones((config['ANALYSIS_PARAMS']['N_STAGES'], 4))
    p_vals_e = np.ones((config['ANALYSIS_PARAMS']['N_STAGES'], 2))
    for p in range(config['ANALYSIS_PARAMS']['N_STAGES']):
        p_vals_l[p, :] = 2 * (
                np.sum(cos_sim_mat_rnd[p, :, :] >= cos_sim_mat[p, :][:, None], axis=-1) / config['ANALYSIS_PARAMS'][
            'N_REPS'])
        p_vals_e[p, :] = 2 * (np.sum(cos_sim_mat_e_rnd[p, :, :] >= cos_sim_mat_e[p, :][:, None], axis=-1) /
                              config['ANALYSIS_PARAMS']['N_REPS'])

    save_data(
        [cos_sim_mat_e, cos_sim_mat, cos_sim_mat_e_rnd, cos_sim_mat_rnd, epochs_task1_e, epochs_task2_e, epochs_task1_l,
         epochs_task2_l, epochs_context1_e, epochs_context2_e, epochs_task1_l_irr, epochs_task2_l_irr],
        ['cos_sim_mat_e', 'cos_sim_mat', 'cos_sim_mat_e_rnd', 'cos_sim_mat_rnd', 'epochs_task1_e', 'epochs_task2_e',
         'epochs_task1_l', 'epochs_task2_l', 'epochs_context1_e', 'epochs_context2_e', 'epochs_task1_l_irr',
         'epochs_task2_l_irr'],
        config['PATHS']['output_path'] + 'exp2_selectivity_dat_early_' +
        str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' +
        str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) +
        '_late_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' +
        str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + '.pickle')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    dat = load_data(config['PATHS']['output_path'] + 'exp2_selectivity_dat_early_' +
    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' +
    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) +
    '_late_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' +
    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + '.pickle')
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

    task_1_e = [np.concatenate([epochs_task1_e[i_stage][:, :1, 0], epochs_context1_e[i_stage][:, :1, 0]], axis=1) for
                i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
    task_2_e = [np.concatenate([epochs_task2_e[i_stage][:, :1, 0], epochs_context2_e[i_stage][:, :1, 0]], axis=1) for
                i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]

    task_1 = [np.concatenate(
        [epochs_task1_l[i_stage][:, :2, 0], epochs_task1_l_irr[i_stage][:, :1, 0], epochs_task1_l[i_stage][:, 2:3, 0]],
        axis=1) for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
    task_2 = [np.concatenate(
        [epochs_task2_l[i_stage][:, :2, 0], epochs_task2_l_irr[i_stage][:, :1, 0], epochs_task2_l[i_stage][:, 2:3, 0]],
        axis=1) for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]

    tasks = [np.array([task_1[i_stage], task_2[i_stage]]) for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
    tasks_e = [np.array([task_1_e[i_stage], task_2_e[i_stage]]) for i_stage in
               range(config['ANALYSIS_PARAMS']['N_STAGES'])]

    cos_sim_rnd = compute_lr_null_cos_sim(tasks, reps=10000)
    cos_sim_e_rnd = compute_lr_null_cos_sim(tasks_e, reps=10000)



if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages')

    taskset = np.array(config['ENCODING_EXP2']['taskset'])
    context = np.array(config['ENCODING_EXP2']['context'])
    shape = np.array(config['ENCODING_EXP2']['shape'])
    width = np.array(config['ENCODING_EXP2']['width'])
    xor = np.array(config['ENCODING_EXP2']['xor'])

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)
    print('Generating decoding null (within stage) ...')
    scores_rnd_early = run_decoding_colour_locked(data_eq, labels_eq,
                                                  config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'],
                                                  factors=[context, taskset], mode='only_null')

    scores_rnd_late = run_decoding_colour_locked(data_eq, labels_eq,
                                                  config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'],
                                                  factors=[context, taskset], mode='only_null')

    dec_data_exp2_rnd = [scores_rnd_early, scores_rnd_late]

    names_rnd = ['scores_rnd_early', 'scores_rnd_late']
    save_data(dec_data_exp2_rnd, names_rnd,
              config['PATHS']['output_path'] + 'exp2_decoding_within_stage_null_early_' +
              str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' +
              str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) +
              '_late_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' +
              str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + '.pickle')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    dat = load_data(config['PATHS']['output_path'] + 'exp2_decoding_within_stage_null_early_' +
                    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' +
                    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) +
                    '_late_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' +
                    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + '.pickle')
    scores_rnd_early = dat['scores_rnd_early']
    scores_rnd_late = dat['scores_rnd_late']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    dec_exp2 = load_data(config['PATHS']['output_path'] + 'exp2_dec_timeres_xor2_'+ str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages' + '_within_stage.pickle')
    dec_time = dec_exp2['dec_time']

#%%
print ('Generating Figure 4... ')
print ('')
# create a grid for figure 3
fig, gs = creat_plot_grid(3, 5, 2, width_ratios=[1,1,1,0.7,1])
x_data_time = np.linspace(-0.2, 1.5, dec_time.shape[0])

tw_early_2 = config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK']
tw = config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK']

patch_pars = [
    {'xy': ((tw_early_2[0] - 50) / 100, 0.0),
     'width': (tw_early_2[1] - tw_early_2[0]) / 100,
     'height': 1},

    {'xy': ((tw[0] - 50) / 100, 0.0),
     'width': (tw[1] - tw[0]) / 100,
     'height': 1},

    {'xy': ((tw[0] - 50) / 100, 0.0),
     'width': (tw[1] - tw[0]) / 100,
     'height': 1}
]
titles = ['context', 'shape', 'xor']
scale_scatter = 0.25
context_ylim = [0.42, 0.75]
xor_ylim = [0.45, 0.72]

y_lims = [context_ylim, xor_ylim, xor_ylim]

vars = [0,2,-2]
for i_panel in range(3):
    ax = line_plot_timevar(gs=gs[i_panel, -2], fig=fig, x=x_data_time, y=dec_time[:, vars[i_panel], 1, :].T[None, :, :],
                           color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel] + ' cross-set gen.',
                           ylim=y_lims[i_panel],
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars[i_panel],
                           xlim=[-0.2, 1.0], if_title=True, if_sem=None, xaxese_booo=True)

print('******* Context coding analysis *******')
plot_dec_xgen(gs[0, -1], fig, scores_context_early, scores_context_rnd_early, title='context coding', y_lim=context_ylim,
              tails=['two', 'greater'], null_within=scores_rnd_early[1,:,:][None,:,:])
plot_scatter(gs=gs[0, 0], fig=fig, x=epochs_task1_e[0][:, 0, 0], y=epochs_task2_e[0][:, 0, 0], scale=scale_scatter,
             xaxis_label='context (set 1)', yaxis_label=('context (set 2)'),reg_null=cos_sim_mat_e_rnd[0, 0, :],
             title='stage 1 selectivity', plot_reg=True, overlay_model=None)
plot_scatter(gs=gs[0, 1], fig=fig, x=epochs_task1_e[-1][:, 0, 0], y=epochs_task2_e[-1][:, 0, 0], scale=scale_scatter,
             xaxis_label='context (set 1)', yaxis_label=('context (set 2)'),reg_null=cos_sim_mat_e_rnd[-1, 0, :],
             title='stage 4 selectivity', plot_reg=True, overlay_model=None)
print ('')
print('******* Shape coding analysis *******')
plot_dec_xgen(gs[1, -1], fig, scores_shape_late, scores_shape_rnd_late, title='shape coding', y_lim=xor_ylim,
              tails=['two', 'greater'], null_within=scores_rnd_late)
plot_scatter(gs=gs[1, 0], fig=fig, x=epochs_task1_l[0][:, 1, 0], y=epochs_task2_l[0][:, 1, 0], scale=scale_scatter,
             xaxis_label='shape (set 1)', yaxis_label=('shape (set 2)'), title='stage 1 selectivity', reg_null=cos_sim_mat_rnd[0, 1, :],
             plot_reg=True, overlay_model=None)
plot_scatter(gs=gs[1, 1], fig=fig, x=epochs_task1_l[-1][:, 1, 0], y=epochs_task2_l[-1][:, 1, 0], scale=scale_scatter,
             xaxis_label='shape (set 1)', yaxis_label=('shape (set 2)'), title='stage 4 selectivity', reg_null=cos_sim_mat_rnd[-1, 1, :],
             plot_reg=True, overlay_model=None)
print ('')
print('******* XOR coding analysis *******')
plot_dec_xgen(gs[2, -1], fig, scores_xor_late, scores_xor_rnd_late, title='xor coding', y_lim=xor_ylim,
              tails=['greater', 'greater'], null_within=scores_rnd_late)
plot_scatter(gs=gs[2, 0], fig=fig, x=epochs_task1_l[0][:, 2, 0], y=epochs_task2_l[0][:, 2, 0], scale=.25,
             xaxis_label='xor (set 1)', yaxis_label=('xor (set 2)'), title='stage 1 selectivity', reg_null=cos_sim_mat_rnd[0, 2, :],
             plot_reg=True, overlay_model=None)
plot_scatter(gs=gs[2, 1], fig=fig, x=epochs_task1_l[-1][:, 2, 0], y=epochs_task2_l[-1][:, 2, 0], scale=.25,
             xaxis_label='xor (set 1)', yaxis_label=('xor (set 2)'), title='stage 4 selectivity', reg_null=cos_sim_mat_rnd[-1, 2, :],
             plot_reg=True, overlay_model=None)


print ('')
print('******* Selectivity alignment analysis *******')
y_lim_r = [-0.3, 0.8]
print('context cross-set correlation')
plot_cos_sim(gs=gs[0,2], fig=fig, obs=cos_sim_mat_e[:, 0], ler_rnd=cos_sim_e_rnd[:, :,  0], rnd_null=cos_sim_mat_e_rnd[:, 0, :], title='selectivity alignment', y_lim=y_lim_r, tail='greater')
print('shape cross-set correlation')
plot_cos_sim(gs=gs[1,2], fig=fig, obs=cos_sim_mat[:, 1], ler_rnd=cos_sim_rnd[:, :, 1], rnd_null=cos_sim_mat_rnd[:, 1, :] , title='selectivity alignment', y_lim=y_lim_r, tail='greater')
print('xor cross-set correlation')
plot_cos_sim(gs=gs[2,2], fig=fig, obs=cos_sim_mat[:, 2], ler_rnd=cos_sim_rnd[:, :, 2], rnd_null= cos_sim_mat_rnd[:, 2, :], title='selectivity alignment', y_lim=y_lim_r, tail='greater')
plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'fig_4_1.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'fig_4_1.png', dpi=300)


fig, gs = creat_plot_grid(3, 2, 2, width_ratios=[1,0.11])
ax, p_vals = plot_bar_epochs_with_null(
    gs=gs[0, -1],
    fig=fig,
    scores=scores_context_early,              # Shape (2, 4) - full matrix
    scores_rnd=scores_rnd_early.transpose((-1,0,1)),      # Shape (500, 2, 4) - full matrix
    title='decoding',
    ylabel='',
    ylim=context_ylim,
    color=['grey', 'black'],                  # Grey=Stage1, Black=Stage4
    cross_gen_idx=1                           # 0=within-set, 1=cross-set
)

ax, p_vals = plot_bar_epochs_with_null(
    gs=gs[1, -1],
    fig=fig,
    scores=scores_shape_late,              # Shape (2, 4) - full matrix
    scores_rnd=scores_rnd_late.transpose((-1,0,1)),      # Shape (500, 2, 4) - full matrix
    title='decoding',
    ylabel='',
    ylim=xor_ylim,
    color=['grey', 'black'],                  # Grey=Stage1, Black=Stage4
    cross_gen_idx=1                           # 0=within-set, 1=cross-set
)

ax, p_vals = plot_bar_epochs_with_null(
    gs=gs[2, -1],
    fig=fig,
    scores=scores_xor_late,              # Shape (2, 4) - full matrix
    scores_rnd=scores_rnd_late.transpose((-1,0,1)),      # Shape (500, 2, 4) - full matrix
    title='decoding',
    ylabel='',
    ylim=xor_ylim,
    color=['grey', 'black'],                  # Grey=Stage1, Black=Stage4
    cross_gen_idx=1                           # 0=within-set, 1=cross-set
)
plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'fig_4_2.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'fig_4_2.png', dpi=300)
print('')
print('Saved to ' + config['PATHS']['out_template_figures'])