from fun_lib import *
with open('config.yml') as file:
    config = yaml.full_load(file)

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
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages')

    dec_time_res = run_time_resolved_dec_exp2(data, labels, if_save=True,
                                              fname='exp2_dec_timeres_xor2_'+ str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages',
                                              mode='within_stage')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    dec_exp2 = load_data(config['PATHS']['output_path'] + 'exp2_dec_timeres_xor2_'+ str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages' + '_within_stage.pickle')
    dec_time_res = dec_exp2['dec_time']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':

    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages')

    taskset = np.array(config['ENCODING_EXP2']['taskset'])
    context = np.array(config['ENCODING_EXP2']['context'])
    shape = np.array(config['ENCODING_EXP2']['shape'])
    width = np.array(config['ENCODING_EXP2']['width'])
    xor = np.array(config['ENCODING_EXP2']['xor'])
    xor2 = np.array(config['ENCODING_EXP2']['xor2'])

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)

    scores_context, scores_set,  scores_shape, scores_width, scores_xor, scores_xor2 = run_decoding_shape_locked(
        data_eq,
        labels_eq,
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'],
        factors=[context,
                 taskset,
                 shape,
                 width,
                 xor, xor2])

    print('Constructing null distribution for decoding analysis...')
    scores_context_rnd, scores_set_rnd, scores_shape_rnd, scores_width_rnd, scores_xor_rnd, scores_xor2_rnd = run_decoding_ler_null_shape_locked(
        data_eq, labels_eq, time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'],
        n_reps=config['ANALYSIS_PARAMS']['N_REPS'],
        factors=[context, taskset, shape, width, xor, xor2])


    dec_data_exp2_late = [scores_context, scores_set, scores_shape, scores_width, scores_xor,
                          scores_xor2,
                          scores_context_rnd, scores_set_rnd, scores_shape_rnd, scores_width_rnd,
                          scores_xor_rnd, scores_xor2_rnd]
    names_late = ['scores_context', 'scores_set', 'scores_shape', 'scores_width', 'scores_xor',
                  'scores_xor2',
                  'scores_context_rnd', 'scores_set_rnd', 'scores_shape_rnd', 'scores_width_rnd',
                  'scores_xor_rnd', 'scores_xor2_rnd']

    save_data(dec_data_exp2_late, names_late,
              config['PATHS']['output_path'] + 'exp2_decoding_time_avg_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) + '.pickle'
              )

    sd = np.array([scores_context, scores_set, scores_xor2]).mean(0)
    sd_rnd = np.array([scores_context_rnd, scores_set_rnd, scores_xor2_rnd]).mean(0)
    dec_list = [scores_set, scores_xor2, scores_context]
    dec_rnd_list = [scores_set_rnd, scores_xor2_rnd, scores_context_rnd]

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    dat_col_locked = load_data(config['PATHS']['output_path'] + 'exp2_decoding_time_avg_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) + '.pickle')

    scores_context = dat_col_locked['scores_context']
    scores_set = dat_col_locked['scores_set']
    scores_xor2 = dat_col_locked['scores_xor2']

    scores_context_rnd = dat_col_locked['scores_context_rnd']
    scores_set_rnd = dat_col_locked['scores_set_rnd']
    scores_xor2_rnd = dat_col_locked['scores_xor2_rnd']

    sd = np.array([scores_context, scores_set, scores_xor2]).mean(0)
    sd_rnd = np.array([scores_context_rnd, scores_set_rnd, scores_xor2_rnd]).mean(0)

    dec_list = [scores_set, scores_xor2, scores_context]
    dec_rnd_list = [scores_set_rnd, scores_xor2_rnd, scores_context_rnd]

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages')


    var_encoding_sel = [config['SEL_ENCODING_EXP2']['context_sel'],
                        config['SEL_ENCODING_EXP2']['set_sel'],
                        ]

    time_window = config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SEL']
    selectivity_coefficients_xval = get_betas_cross_val_exp2(data=data,
                                                             labels=labels,
                                                             condition_labels=var_encoding_sel,
                                                             normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                                             time_window=time_window,
                                                             )

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

    selectivity_coefficients = [selectivity_coefficients_xval[_].mean(-1) for _ in range(config['ANALYSIS_PARAMS']['N_STAGES'])]

    p_vals_rnd = p_vals_rnd[0]
    p_vals_str = p_vals_str[0]

    save_data([selectivity_coefficients, rnd_matrix, p_vals_rnd, str_matrix, p_vals_str], ['selectivity_coefficients', 'rnd_matrix', 'p_vals_rnd', 'str_matrix', 'p_vals_str'],
              config['PATHS']['output_path'] + 'selectivity_coefficients_exp2_' + str(time_window[0]) + '_' + str(time_window[1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' '.pickle')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    sel_data_exp2 = load_data(config['PATHS']['output_path'] + 'selectivity_coefficients_exp2_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SEL'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SEL'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages.pickle')
    rnd_matrix = sel_data_exp2['rnd_matrix']
    p_vals_rnd = sel_data_exp2['p_vals_rnd']
    str_matrix = sel_data_exp2['str_matrix']
    p_vals_str = sel_data_exp2['p_vals_str']
    selectivity_coefficients = sel_data_exp2['selectivity_coefficients']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':

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

    if config['ANALYSIS_PARAMS']['N_STAGES'] < 5:
        ses_all = combine_session_lists(mode='stages', which_exp='exp2')
    else:
        ses_all = combine_session_lists(mode='sliding_window', which_exp='exp2')

    fix_breaks_animals = get_fixation_breaks([ses_all], experiment_label='exp2')[0]
    no_reward_fix = fix_breaks_animals[:, 2] / fix_breaks_animals[:, 3]
    reward_fix = fix_breaks_animals[:, 1] / fix_breaks_animals[:, 3]
    dat = no_reward_fix / reward_fix
    data_prop_s_exp2_set2 = np.array_split(dat, config['ANALYSIS_PARAMS']['N_STAGES'])

    save_data([dat_prop, data_prop_s_exp2_set2], ['dat_prop', 'data_prop_s_exp2_set2'], config['PATHS']['output_path'] + 'fixation_breaks_prop_exp2_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages.pickle')
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    beh_data_exp2 = load_data(config['PATHS']['output_path'] + 'fixation_breaks_prop_exp2_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages.pickle')
    data_prop_s_exp2_set2 = beh_data_exp2['data_prop_s_exp2_set2']
    dat_prop = beh_data_exp2['dat_prop']

#%%

print ('Generating Figure 3... ')
print ('')
# create a grid for figure 3
fig, gs = creat_plot_grid(2, 4, 2)
x_data_time = np.linspace(-0.2, 1.5, dec_time_res.shape[0])

tw = config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK']
patch_pars = {'xy': ((tw[0] - 50) / 100, 0.0),
     'width': (tw[1] - tw[0]) / 100,
     'height': 1}
titles = ['stimulus set', 'xor 2 (set*context)', 'context']
tails_lis_dec = [-1, -1, 0]

# plot the proportion of fixation break between rewarded and non-rewarded trials
plot_mean_and_ci_prop(gs=gs[0,2], fig=fig, reward_prop=data_prop_s_exp2_set2, n_perm=10000, baseline_val=1.0, vmin=0, title="trial terminations")
# compute two-sample t-test and pvalue for stage one prop data
plot_data_comparison(gs[0,3], fig, dat_prop[0], dat_prop[1], title='terminations: exp1 vs exp2', ylim=[0, 6])
print ('')
i_panel=1
var_indinces = [1, -1, 0]
for i_panel in range(3):
    ax = line_plot_timevar(gs=gs[1, i_panel], fig=fig, x=x_data_time, y=dec_time_res[:, var_indinces[i_panel], 0, :].T[None, :, :],
                           color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel] + ' decoding', ylim=[0.45, 0.8],
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars,
                           xlim=[-0.2, 1.0], if_title=True, if_sem=None, xaxese_booo=True)
    print('******* ' + titles[i_panel] + ' decoding analysis *******')
    plot_significance_stars(ax, dec_list[i_panel], dec_rnd_list[i_panel],
                            time_window=tw, ylim=[0.45, 0.8], tail='greater')
    print('')



sd_colour_dec = np.array([dec_time_res[:,0,0,:], dec_time_res[:,1,0,:], dec_time_res[:,-1,0,:]]).mean(0)
ax = line_plot_timevar(gs=gs[1, -1], fig=fig, x=x_data_time,
                       y=sd_colour_dec.T[None, :, :],
                       color=['grey', 'black'],
                       xlabel='time (s)', ylabel='accuracy', title='dimensionality',
                       ylim=[0.45, 0.8],
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                       patch_pars=patch_pars,
                       xlim=[-0.2, 1.0], if_title=True, if_sem=None, xaxese_booo=True)
print('******* Dimensionality analysis *******')
plot_significance_stars(ax, sd, sd_rnd,
                        time_window=tw, ylim=[0.45, 0.8], tail='greater')
print('')
plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'fig_3_1.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'fig_3_1.png', dpi=300)

fig, gs = creat_plot_grid(4, 4, 2)
# first 2 rows
x_data_r2 = [selectivity_coefficients[0][:, config['ANALYSIS_PARAMS']['COLOUR_ID']], selectivity_coefficients[0][:, config['ANALYSIS_PARAMS']['COLOUR_ID']],
             selectivity_coefficients[1][:, config['ANALYSIS_PARAMS']['COLOUR_ID']], selectivity_coefficients[1][:, config['ANALYSIS_PARAMS']['COLOUR_ID']]]
y_data_r2 = [selectivity_coefficients[0][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], selectivity_coefficients[0][:, -1],
             selectivity_coefficients[1][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], selectivity_coefficients[1][:, -1]]
x_data_r3 = [selectivity_coefficients[0][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], None, selectivity_coefficients[1][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], None]
y_data_r3 = [selectivity_coefficients[0][:, -1], None, selectivity_coefficients[1][:, -1], None]

xaxis_label_lis_r2 = ['set', 'set', 'set', 'set']
yaxis_label_lis_r2 = ['set*context', 'context', 'set*context', 'context']
xaxis_label_lis_r3 = ['set*context', None, 'set*context', None]
yaxis_label_lis_r3 = ['context', None, 'context', None]
cov_dat = [None, selectivity_coefficients[0], None, selectivity_coefficients[1]]

for i_panel in range(4):
    plot_scatter(gs=gs[0, i_panel], fig=fig, x=x_data_r2[i_panel], y=y_data_r2[i_panel], scale=.2,
                 xaxis_label=xaxis_label_lis_r2[i_panel], yaxis_label=yaxis_label_lis_r2[i_panel],
                 overlay_model='contour', kde_smooth=1.2, out_factor=1.25)
    # if i_panel is odd
    if i_panel % 2 == 1:
        plot_cov(gs=gs[1, i_panel], fig=fig, dat=cov_dat[i_panel], scale=0.006, labels=['set', 's*c', 'cxt'], title='covariance')

    if i_panel % 2 == 0:
        plot_scatter(gs=gs[1, i_panel], fig=fig, x=x_data_r3[i_panel], y=y_data_r3[i_panel], scale=.2,
                     xaxis_label=xaxis_label_lis_r3[i_panel], yaxis_label=yaxis_label_lis_r3[i_panel],
                     overlay_model='contour',kde_smooth=1.2, out_factor=1.25)


#second two rows
x_data_r2 = [selectivity_coefficients[2][:, config['ANALYSIS_PARAMS']['COLOUR_ID']], selectivity_coefficients[2][:, config['ANALYSIS_PARAMS']['COLOUR_ID']],
             selectivity_coefficients[-1][:, config['ANALYSIS_PARAMS']['COLOUR_ID']], selectivity_coefficients[-1][:, config['ANALYSIS_PARAMS']['COLOUR_ID']]]
y_data_r2 = [selectivity_coefficients[2][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], selectivity_coefficients[2][:, -1],
             selectivity_coefficients[-1][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], selectivity_coefficients[-1][:, -1]]
x_data_r3 = [selectivity_coefficients[2][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], None, selectivity_coefficients[-1][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], None]
y_data_r3 = [selectivity_coefficients[2][:, -1], None, selectivity_coefficients[-1][:, -1], None]

xaxis_label_lis_r2 = ['set', 'set', 'set', 'set']
yaxis_label_lis_r2 = ['set*context', 'context', 'set*context', 'context']
xaxis_label_lis_r3 = ['set*context', None, 'set*context', None]
yaxis_label_lis_r3 = ['context', None, 'context', None]
cov_dat = [None, selectivity_coefficients[2], None, selectivity_coefficients[-1]]

for i_panel in range(4):
    plot_scatter(gs=gs[2, i_panel], fig=fig, x=x_data_r2[i_panel], y=y_data_r2[i_panel], scale=.2,
                 xaxis_label=xaxis_label_lis_r2[i_panel], yaxis_label=yaxis_label_lis_r2[i_panel],
                 overlay_model='contour', kde_smooth=1.2, out_factor=1.25)
    # if i_panel is odd
    if i_panel % 2 == 1:
        plot_cov(gs=gs[3, i_panel], fig=fig, dat=cov_dat[i_panel], scale=0.006, labels=['set', 's*c', 'cxt'], title='covariance')

    if i_panel % 2 == 0:
        plot_scatter(gs=gs[3, i_panel], fig=fig, x=x_data_r3[i_panel], y=y_data_r3[i_panel], scale=.2,
                     xaxis_label=xaxis_label_lis_r3[i_panel], yaxis_label=yaxis_label_lis_r3[i_panel],
                     overlay_model='contour', kde_smooth=1.2, out_factor=1.25)
plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'fig_3_2.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'fig_3_2.png', dpi=300)


fig, gs = creat_plot_grid(1, 4, 2)
print('******* Distance from random analysis *******')
plot_distance(gs[0, :2], fig, rnd_matrix, 'distance from random', pvals=p_vals_rnd)
print('')
print('******* Distance from minimal analysis *******')
plot_distance(gs[0, 2:], fig, str_matrix, 'distance from structured', pvals=p_vals_str)
print('')
plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'fig_3_3.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'fig_3_3.png', dpi=300)
print('')
print('Saved to ' + config['PATHS']['out_template_figures'])