from fun_lib import *

with open('config.yml') as file:
    config = yaml.full_load(file)

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time')

    taskset = np.array(config['ENCODING_EXP2']['taskset'])
    context = np.array(config['ENCODING_EXP2']['context'])
    shape = np.array(config['ENCODING_EXP2']['shape'])
    width = np.array(config['ENCODING_EXP2']['width'])
    xor = np.array(config['ENCODING_EXP2']['xor'])

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)
    print('Running decoding analysis...')
    scores_context_early, scores_set_early = run_decoding_colour_locked(data_eq, labels_eq,
                                                                        config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'],
                                                                        factors=[context, taskset])
    scores_context_late, scores_set_late, scores_shape_late, scores_width_late, scores_xor_late = run_decoding_shape_locked(
        data_eq,
        labels_eq,
        config[
            'ANALYSIS_PARAMS'][
            'TIME_WINDOW'],
        factors=[context,
                 taskset,
                 shape,
                 width,
                 xor])

    print('Constructing null distribution for decoding analysis...')
    scores_context_rnd_early, scores_set_rnd_early = run_decoding_ler_null_colour_locked(data_eq, labels_eq,
                                                                                         time_window=
                                                                                         config['ANALYSIS_PARAMS'][
                                                                                             'TIME_WINDOW_EARLY'],
                                                                                         n_reps=
                                                                                         config['ANALYSIS_PARAMS'][
                                                                                             'N_REPS'],
                                                                                         factors=[context, taskset])

    scores_context_rnd_late, scores_set_rnd_late, scores_shape_rnd_late, scores_width_rnd_late, scores_xor_rnd_late = run_decoding_ler_null_shape_locked(
        data_eq, labels_eq, time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW'],
        n_reps=config['ANALYSIS_PARAMS']['N_REPS'],
        factors=[context, taskset, shape, width, xor])

    dec_data_exp2_early = [scores_context_early, scores_set_early, scores_context_rnd_early, scores_set_rnd_early]
    names_early = ['scores_context_early', 'scores_set_early', 'scores_context_rnd_early', 'scores_set_rnd_early']
    save_data(dec_data_exp2_early, names_early,
              config['PATHS']['output_path'] + 'exp2_decoding_collocked_' + str(
                  config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'][0]) + '_' + str(
                  config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'][1]) + '.pickle')

    dec_data_exp2_late = [scores_context_late, scores_set_late, scores_shape_late, scores_width_late, scores_xor_late,
                          scores_context_rnd_late, scores_set_rnd_late, scores_shape_rnd_late, scores_width_rnd_late,
                          scores_xor_rnd_late]
    names_late = ['scores_context_late', 'scores_set_late', 'scores_shape_late', 'scores_width_late', 'scores_xor_late',
                  'scores_context_rnd_late', 'scores_set_rnd_late', 'scores_shape_rnd_late', 'scores_width_rnd_late',
                  'scores_xor_rnd_late']
    save_data(dec_data_exp2_late, names_late,
              config['PATHS']['output_path'] + 'exp2_decoding_shapelocked_' + str(
                  config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(
                  config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.pickle')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
    dat_col_locked = load_data(config['PATHS']['output_path'] + 'exp2_decoding_collocked_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'][1]) + '.pickle')
    dat_shape_locked = load_data(config['PATHS']['output_path'] + 'exp2_decoding_shapelocked_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.pickle')

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

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':
    var_encoding_sel = [config['SEL_ENCODING_EXP1']['colour_sel'],
                        config['SEL_ENCODING_EXP1']['shape_sel'],
                        config['SEL_ENCODING_EXP1']['xor_sel'],
                        config['SEL_ENCODING_EXP1']['width_sel'],
                        config['SEL_ENCODING_EXP1']['int_irrel_sel']]

    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time')

    epochs_task1_l, epochs_task1_l_irr = get_betas_cross_val_2(data=data,
                                                               labels=labels,
                                                               condition_labels=var_encoding_sel,
                                                               normalisation=config['ANALYSIS_PARAMS'][
                                                                   'NORMALISATION_SEL'],
                                                               time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW'],
                                                               task='task_1',
                                                               )

    epochs_task2_l, epochs_task2_l_irr = get_betas_cross_val_2(data=data,
                                                               labels=labels,
                                                               condition_labels=var_encoding_sel,
                                                               normalisation=config['ANALYSIS_PARAMS'][
                                                                   'NORMALISATION_SEL'],
                                                               time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW'],
                                                               task='task_2',
                                                               )

    epochs_task1_e, _ = get_betas_cross_val_2(data=data,
                                              labels=labels,
                                              condition_labels=var_encoding_sel,
                                              normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                              time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'],
                                              task='task_1',
                                              )

    epochs_task2_e, _ = get_betas_cross_val_2(data=data,
                                              labels=labels,
                                              condition_labels=var_encoding_sel,
                                              normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                              time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'],
                                              task='task_2',
                                              )

    epochs_context1_e, _ = get_betas_cross_val_2(data=data,
                                                 labels=labels,
                                                 condition_labels=var_encoding_sel,
                                                 normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                                 time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'],
                                                 task='context_1',
                                                 )

    epochs_context2_e, _ = get_betas_cross_val_2(data=data,
                                                 labels=labels,
                                                 condition_labels=var_encoding_sel,
                                                 normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                                 time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'],
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
        config['PATHS']['output_path'] + 'exp2_selectivity_dat.pickle')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
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

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'run':
    context = np.array(config['ENCODING_EXP1']['colour'])
    shape = np.array(config['ENCODING_EXP1']['shape'])
    width = np.array(config['ENCODING_EXP1']['width'])
    xor = np.array(config['ENCODING_EXP1']['xor'])

    data1, labels1 = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time')
    data2, labels2 = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_raw')

    # early time window
    context_dec_set1, context_dec_set2, context_dec_early_rnd, p_val_context_early = run_dec_comparison_tasksets(data1,
                                                                                                                 labels1,
                                                                                                                 data2,
                                                                                                                 labels2,
                                                                                                                 time_window=
                                                                                                                 config[
                                                                                                                     'ANALYSIS_PARAMS'][
                                                                                                                     'TIME_WINDOW_EARLY'],
                                                                                                                 variables=[
                                                                                                                     context],
                                                                                                                 tails=[
                                                                                                                     'greater'])

    # late time window
    vars_dec_set1, vars_dec_set2, vars_dec_rnd, p_val_vars = run_dec_comparison_tasksets(data1, labels1, data2,
                                                                                         labels2, time_window=
                                                                                         config[
                                                                                             'ANALYSIS_PARAMS'][
                                                                                             'TIME_WINDOW'],
                                                                                         variables=[xor, shape,
                                                                                                    context],
                                                                                         tails=['greater', 'greater',
                                                                                                'greater']
                                                                                         )
    save_data([context_dec_set1, context_dec_set2, context_dec_early_rnd, p_val_context_early, vars_dec_set1,
               vars_dec_set2, vars_dec_rnd, p_val_vars],
              ['context_dec_set1', 'context_dec_set2', 'context_dec_early_rnd', 'p_val_context_early', 'vars_dec_set1',
               'vars_dec_set2', 'vars_dec_rnd', 'p_val_vars'],
              config['PATHS']['output_path'] + 'exp2_decoding_set_comparison.pickle')
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_TIME'] == 'observe':
    dat = load_data(config['PATHS']['output_path'] + 'exp2_decoding_set_comparison.pickle')
    context_dec_set1 = dat['context_dec_set1']
    context_dec_set2 = dat['context_dec_set2']
    context_dec_early_rnd = dat['context_dec_early_rnd']
    p_val_context_early = dat['p_val_context_early']
    vars_dec_set1 = dat['vars_dec_set1']
    vars_dec_set2 = dat['vars_dec_set2']
    vars_dec_rnd = dat['vars_dec_rnd']
    p_val_vars = dat['p_val_vars']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time')

    taskset = np.array(config['ENCODING_EXP2']['taskset'])
    context = np.array(config['ENCODING_EXP2']['context'])
    shape = np.array(config['ENCODING_EXP2']['shape'])
    width = np.array(config['ENCODING_EXP2']['width'])
    xor = np.array(config['ENCODING_EXP2']['xor'])

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)
    print('Generating decoding null (within stage) ...')
    scores_rnd_early = run_decoding_colour_locked(data_eq, labels_eq,
                                                  config['ANALYSIS_PARAMS']['TIME_WINDOW_EARLY'],
                                                  factors=[context, taskset], mode='only_null')

    scores_rnd_late = run_decoding_colour_locked(data_eq, labels_eq,
                                                  config['ANALYSIS_PARAMS']['TIME_WINDOW'],
                                                  factors=[context, taskset], mode='only_null')

    dec_data_exp2_rnd = [scores_rnd_early, scores_rnd_late]

    names_rnd = ['scores_rnd_early', 'scores_rnd_late']
    save_data(dec_data_exp2_rnd, names_rnd,
              config['PATHS']['output_path'] + 'exp2_decoding_within_stage_null.pickle')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    dat = load_data(config['PATHS']['output_path'] + 'exp2_decoding_within_stage_null.pickle')
    scores_rnd_early = dat['scores_rnd_early']
    scores_rnd_late = dat['scores_rnd_late']


# create a grid for figure 3
fig, gs = creat_plot_grid(2, 4, 2)
plot_dec_xgen(gs[0, 0], fig, scores_context_early, scores_context_rnd_early, title='context coding', y_lim=[0.35, 0.8],
              tails=['two', 'greater'], null_within=scores_rnd_early)
plot_scatter(gs=gs[0, 1], fig=fig, x=epochs_task1_e[0][:, 0, 0], y=epochs_task2_e[0][:, 0, 0], scale=.25,
             xaxis_label='colour 1vs2 (set 1)\nselectivity', yaxis_label=('colour 3vs4 (set 2)\nselectivity'),
             title='stage 1', plot_reg=True)
plot_scatter(gs=gs[0, 2], fig=fig, x=epochs_task1_e[-1][:, 0, 0], y=epochs_task2_e[-1][:, 0, 0], scale=.25,
             xaxis_label='colour 1vs2 (set 1)\nselectivity', yaxis_label=('colour 3vs4 (set 2)\nselectivity'),
             title='stage 4', plot_reg=True)
plot_decoding_set_comp(gs=gs[0, 3], fig=fig, set1_dec=context_dec_set1[0, :], set2_dec=context_dec_set2[0, :],
                       p_val=p_val_context_early[0], y_lim=[0.45, 0.8],
                       title='colour decoding\n(set 1 vs set 2)')
print('Colour decoding set 1 vs set 2 over learning p-value = ', str(round(p_val_context_early[0], 3)))

plot_dec_xgen(gs[1, 0], fig, scores_set_early, scores_set_rnd_early, title='set coding', y_lim=[0.35, 0.8],
              tails=['smaller', 'smaller'], null_within=scores_rnd_early)
plot_scatter(gs=gs[1, 1], fig=fig, x=epochs_context1_e[0][:, 0, 0], y=epochs_context2_e[0][:, 0, 0], scale=.25,
             xaxis_label='set (colour 1vs3)\nselectivity', yaxis_label=('set (colour 2vs4)\nselectivity'),
             title='stage 1', plot_reg=True)
plot_scatter(gs=gs[1, 2], fig=fig, x=epochs_context1_e[-1][:, 0, 0], y=epochs_context2_e[-1][:, 0, 0], scale=.25,
             xaxis_label='set (colour 1vs3)\nselectivity', yaxis_label=('set (colour 2vs4)\nselectivity'),
             title='stage 4', plot_reg=True)
plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_figure_4_' + str(
    config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.svg')
plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_figure_4_' + str(
    config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.png',
            dpi=300)

# create a grid for figure 5
fig, gs = creat_plot_grid(3, 4, 2)
plot_dec_xgen(gs[0, 0], fig, scores_xor_late, scores_xor_rnd_late, title='xor coding', y_lim=[0.35, 0.8],
              tails=['greater', 'greater'], null_within=scores_rnd_late)
plot_scatter(gs=gs[0, 1], fig=fig, x=epochs_task1_l[0][:, 2, 0], y=epochs_task2_l[0][:, 2, 0], scale=.25,
             xaxis_label='xor (set 1)\nselectivity', yaxis_label=('xor (set 2)\nselectivity'), title='stage 1',
             plot_reg=True)
plot_scatter(gs=gs[0, 2], fig=fig, x=epochs_task1_l[-1][:, 2, 0], y=epochs_task2_l[-1][:, 2, 0], scale=.25,
             xaxis_label='xor (set 1)\nselectivity', yaxis_label=('xor (set 2)\nselectivity'), title='stage 4',
             plot_reg=True)
plot_decoding_set_comp(gs=gs[0, 3], fig=fig, set1_dec=vars_dec_set1[0, :], set2_dec=vars_dec_set2[0, :],
                       p_val=p_val_vars[0], y_lim=[0.45, 0.8],
                       title='xor decoding\n(set 1 vs set 2)')
print('xor decoding set 1 vs set 2 over learning p-value = ', str(round(p_val_vars[0], 3)))

plot_dec_xgen(gs[1, 0], fig, scores_shape_late, scores_shape_rnd_late, title='shape coding', y_lim=[0.35, 0.8],
              tails=['two', 'greater'], null_within=scores_rnd_late)
plot_scatter(gs=gs[1, 1], fig=fig, x=epochs_task1_l[0][:, 1, 0], y=epochs_task2_l[0][:, 1, 0], scale=.25,
             xaxis_label='shape (set 1)\nselectivity', yaxis_label=('xor (set 2)\nselectivity'), title='stage 1',
             plot_reg=True)
plot_scatter(gs=gs[1, 2], fig=fig, x=epochs_task1_l[-1][:, 1, 0], y=epochs_task2_l[-1][:, 1, 0], scale=.25,
             xaxis_label='shape (set 1)\nselectivity', yaxis_label=('xor (set 2)\nselectivity'), title='stage 4',
             plot_reg=True)
plot_decoding_set_comp(gs=gs[1, 3], fig=fig, set1_dec=vars_dec_set1[1, :], set2_dec=vars_dec_set2[1, :],
                       p_val=p_val_vars[1], y_lim=[0.45, 0.8],
                       title='shape decoding\n(set 1 vs set 2)')
print('shape decoding set 1 vs set 2 over learning p-value = ', str(round(p_val_vars[1], 3)))

plot_dec_xgen(gs[2, 0], fig, scores_context_late, scores_context_rnd_late, title='context coding', y_lim=[0.35, 0.8],
              tails=['two', 'greater'], null_within=scores_rnd_late)
plot_scatter(gs=gs[2, 1], fig=fig, x=epochs_task1_l[0][:, 0, 0], y=epochs_task2_l[0][:, 0, 0], scale=.25,
             xaxis_label='context (set 1)\nselectivity', yaxis_label=('context (set 2)\nselectivity'), title='stage 1',
             plot_reg=True)
plot_scatter(gs=gs[2, 2], fig=fig, x=epochs_task1_l[-1][:, 0, 0], y=epochs_task2_l[-1][:, 0, 0], scale=.25,
             xaxis_label='context (set 1)\nselectivity', yaxis_label=('context (set 2)\nselectivity'), title='stage 4',
             plot_reg=True)
plot_decoding_set_comp(gs=gs[2, 3], fig=fig, set1_dec=vars_dec_set1[2, :], set2_dec=vars_dec_set2[2, :],
                       p_val=p_val_vars[2], y_lim=[0.45, 0.8],
                       title='colour decoding\n(set 1 vs set 2)')
print('colour decoding set 1 vs set 2 over learning p-value = ', str(round(p_val_vars[2], 3)))
plt.tight_layout()

plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_figure_5_' + str(
    config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.svg')
plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_figure_5_' + str(
    config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.png',
            dpi=300)


