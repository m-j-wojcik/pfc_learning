from fun_lib import *
with open('config.yml') as file:
    config = yaml.full_load(file)

os.makedirs(config['PATHS']['out_template_figures'], exist_ok=True)
os.makedirs(config['PATHS']['output_path'], exist_ok=True)

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_PREPROCESSING'] == 'run':
    if config['ANALYSIS_PARAMS']['N_STAGES'] < 5:
        ses_all = combine_session_lists(mode='stages', which_exp='exp1', combine_all=False)
    else:
        # when discretising learning into more than 4 stages use a sliding window approach to maintain sufficient statistical
        # power (have at least ~70 neurons per stage)
        ses_all = combine_session_lists(mode='sliding_window', which_exp='exp1', combine_all=False)

    data, labels = get_data_stages(observe_or_run='run', session_list=ses_all,
                                   file_name='exp1_f_rates_ses_time_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages', return_data=True)


# compute the proportion of fixation breaks between rewarded and non-rewarded trials as a function of learning
if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Running behavioural analysis')
    if config['ANALYSIS_PARAMS']['N_STAGES'] < 5:
        ses_all = combine_session_lists(mode='stages', which_exp='exp1', combine_all=False)
    else:
        # when discretising learning into more than 4 stages use a sliding window approach to maintain sufficient statistical
        # power (have at least ~70 neurons per stage)
        ses_all = combine_session_lists(mode='sliding_window', which_exp='exp1', combine_all=False)
    fix_breaks_animals = get_fixation_breaks(ses_all, experiment_label='exp1')
    data_prop_s = []
    for i_stage in range(config['ANALYSIS_PARAMS']['N_STAGES']):
        no_reward_fix = fix_breaks_animals[i_stage][:, 2] / fix_breaks_animals[i_stage][:, 3]
        reward_fix = fix_breaks_animals[i_stage][:, 1] / fix_breaks_animals[i_stage][:, 3]
        dat = no_reward_fix / reward_fix
        data_prop_s.append(dat)
    save_data([data_prop_s], ['data_prop_s'], config['PATHS']['output_path'] + 'exp1_beh_terminations' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages' + '.pickle')
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_prop_s = load_data(config['PATHS']['output_path'] + 'exp1_beh_terminations' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages' '.pickle')['data_prop_s']

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Run a time-resolved decoding analysis')
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_time_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages')
    shattering_dim_time, decoding_time, decoding_null_time = run_time_resolved_decoding(data, labels, if_save=True, fname='exp1_decoding_timeres_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages', mode='within_stage')

    shattering_dim_time = shattering_dim_time.mean(-1).T
    decoding_time = decoding_time.T

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_timeres_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '_within_stage.pickle')
    shattering_dim_time = data_exp1['shattering_dim_time'].mean(-1).T
    decoding_time = data_exp1['decoding_time'].T


if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Computing selectivity coefficients')
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_time_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages')




    var_encoding_sel = [config['SEL_ENCODING_EXP1']['colour_sel'],
                        config['SEL_ENCODING_EXP1']['shape_sel'],
                        config['SEL_ENCODING_EXP1']['xor_sel'],
                        config['SEL_ENCODING_EXP1']['width_sel'],
                        config['SEL_ENCODING_EXP1']['int_irrel_sel']]


    selectivity_coefficients_xval, _ = get_betas_cross_val_2(data=data,
                                                             labels=labels,
                                                             condition_labels=var_encoding_sel,
                                                             normalisation=config['ANALYSIS_PARAMS']['NORMALISATION_SEL'],
                                                             time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'],
                                                             task=config['ANALYSIS_PARAMS']['WHICH_TASK_SEL'],
                                                             )


    save_data([selectivity_coefficients_xval], ['selectivity_coefficients_xval'],
              config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][1])  + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '.pickle')
    selectivity_coefficients = [selectivity_coefficients_xval[_].mean(-1) for _ in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][1]) + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '.pickle')
    selectivity_coefficients_xval = data_exp1['selectivity_coefficients_xval']
    selectivity_coefficients = [selectivity_coefficients_xval[_].mean(-1) for _ in range(config['ANALYSIS_PARAMS']['N_STAGES'])]


if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Computing similarity to structured/random models')
    data_exp1 = load_data(config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][1]) + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')
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
              config['PATHS']['output_path'] + 'distance_from_random_exp1_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][1]) + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '.pickle')
    p_vals_rnd = p_vals_rnd[0]
    p_vals_str = p_vals_str[0]

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'distance_from_random_exp1_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][1]) + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '.pickle')
    rnd_matrix = data_exp1['rnd_matrix']
    p_vals_rnd = data_exp1['p_vals_rnd'][0]
    str_matrix = data_exp1['str_matrix']
    p_vals_str = data_exp1['p_vals_str'][0]

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_time_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages')


    variables = [np.array(config['ENCODING_EXP1']['colour']),
                 np.array(config['ENCODING_EXP1']['shape']),
                 np.array(config['ENCODING_EXP1']['width']),
                 np.array(config['ENCODING_EXP1']['xor'])]

    labels = correct_labels_for_cross_gen(labels)
    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'], n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=98)

    print('Running the time-averaged decoding analysis: shape-locked')
    late_shattering_dim, late_decoding, late_xgen_decoding, late_decoding_null, late_xgen_null = run_moving_window_decoding(data_eq, labels_eq, variables,
                                                                                        time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'],
                                                                                        method='svm', n_reps=1,
                                                                                        if_null=True)
    late_shattering_dim_ler_rnd, late_decoding_ler_rnd, late_xgen_decoding_ler_rnd = run_moving_window_decoding_ler_null(data_eq[0], data_eq[-1],
                                                                                              labels_eq[0], variables,
                                                                                              time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'],
                                                                                              n_reps=config['ANALYSIS_PARAMS']['N_REPS'])

    late_data_exp1 = [late_shattering_dim, late_decoding, late_xgen_decoding, late_decoding_null, late_xgen_null, late_shattering_dim_ler_rnd, late_decoding_ler_rnd,
                 late_xgen_decoding_ler_rnd]
    late_names_exp1 = ['late_shattering_dim', 'late_decoding', 'late_xgen_decoding', 'late_decoding_null', 'late_xgen_null', 'late_shattering_dim_ler_rnd', 'late_decoding_ler_rnd',
                  'late_xgen_decoding_ler_rnd']
    save_data(late_data_exp1, late_names_exp1,
              config['PATHS']['output_path'] + 'exp1_decoding_shapelocked_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')

    print('Running the time-averaged decoding analysis: colour-locked')
    early_shattering_dim, early_decoding, early_xgen_decoding, early_decoding_null, early_xgen_null = run_moving_window_decoding(data_eq, labels_eq, variables,
                                                                                        time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'],
                                                                                        method='svm', n_reps=1,
                                                                                        if_null=True)
    early_shattering_dim_ler_rnd, early_decoding_ler_rnd, early_xgen_decoding_ler_rnd = run_moving_window_decoding_ler_null(data_eq[0], data_eq[-1],
                                                                                              labels_eq[0], variables,
                                                                                              time_window=config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'],
                                                                                              n_reps=config['ANALYSIS_PARAMS']['N_REPS'])

    early_data_exp1 = [early_shattering_dim, early_decoding, early_xgen_decoding, early_decoding_null, early_xgen_null, early_shattering_dim_ler_rnd, early_decoding_ler_rnd,
                 early_xgen_decoding_ler_rnd]
    early_names_exp1 = ['early_shattering_dim', 'early_decoding', 'early_xgen_decoding', 'early_decoding_null', 'early_xgen_null', 'early_shattering_dim_ler_rnd', 'early_decoding_ler_rnd',
                  'early_xgen_decoding_ler_rnd']
    save_data(early_data_exp1, early_names_exp1,
              config['PATHS']['output_path'] + 'exp1_decoding_collocked_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')

    decoding_list = [early_decoding[0,:][None,:], late_decoding[1,:][None,:], late_decoding[2,:][None,:], late_decoding[3,:][None,:]]
    decoding_list_rnd = [early_decoding_ler_rnd[:,:,0][:,None,:], late_decoding_ler_rnd[:,:,1][:,None,:], late_decoding_ler_rnd[:,:,2][:,None,:], late_decoding_ler_rnd[:,:,3][:,None,:]]

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    late_data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_shapelocked_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')
    early_data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_collocked_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')

    late_decoding = late_data_exp1['late_decoding']
    late_decoding_ler_rnd = late_data_exp1['late_decoding_ler_rnd']

    late_shattering_dim = late_data_exp1['late_shattering_dim']
    late_shattering_dim_ler_rnd = late_data_exp1['late_shattering_dim_ler_rnd']

    early_decoding = early_data_exp1['early_decoding']
    early_decoding_ler_rnd = early_data_exp1['early_decoding_ler_rnd']

    decoding_list = [early_decoding[:, 0][None,:], late_decoding[:, 1][None,:], late_decoding[:, 2][None,:], late_decoding[:, 3][None,:]]
    decoding_list_rnd = [early_decoding_ler_rnd[:,:,0][:,None,:], late_decoding_ler_rnd[:,:,1][:,None,:], late_decoding_ler_rnd[:,:,2][:,None,:], late_decoding_ler_rnd[:,:,3][:,None,:]]
    sd_decoding = late_shattering_dim.mean(-1)
    sd_decoding_rnd = late_shattering_dim_ler_rnd.mean(-1)

#%%

print ('Generating Figure 2... ')
print ('')
# create a grid for figure 3
fig, gs = creat_plot_grid(5, 4, 2)

# plot the proportion of fixation break between rewarded and non-rewarded trials
plot_mean_and_ci_prop(gs[0], fig, data_prop_s, n_perm=10000)

# prepare decoding data for next three panels
x_data_time = np.linspace(-0.2, 1.5, decoding_time.shape[-1])


tw_colour = config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK']
tw_shape = config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK']

tws=[tw_colour,tw_shape,tw_shape,tw_shape]
patch_pars = [
    {'xy': ((tw_colour[0] - 50) / 100, 0.0),
     'width': (tw_colour[1] - tw_colour[0]) / 100,
     'height': 1},

    {'xy': ((tw_shape[0] - 50) / 100, 0.0),
     'width': (tw_shape[1] - tw_shape[0]) / 100,
     'height': 1},

    {'xy': ((tw_shape[0] - 50) / 100, 0.0),
     'width': (tw_shape[1] - tw_shape[0]) / 100,
     'height': 1},

    {'xy': ((tw_shape[0] - 50) / 100, 0.0),
     'width': (tw_shape[1] - tw_shape[0]) / 100,
     'height': 1}
]

titles = ['colour decoding', 'shape decoding', 'width decoding', 'xor decoding', 'shattering dimensionality']
y_lims = [0.45, 0.95]

# plot time-resolved decoding accuracy
for i_panel in range(3):
    skip_width = 0
    if i_panel == 2:
        skip_width = 1
    ax = line_plot_timevar(gs=gs[i_panel+1], fig=fig, x=x_data_time, y=decoding_time[i_panel+skip_width, :, :][None, :, :], color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel+skip_width], ylim=y_lims,
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars[i_panel+skip_width],
                           xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
    print('******* ' + titles[i_panel+skip_width] + ' analysis *******')
    plot_significance_stars(ax, decoding_list[i_panel+skip_width], decoding_list_rnd[i_panel+skip_width],
                            time_window=tws[i_panel+skip_width], ylim=[0.45, 0.8], tail='greater', y_position=0.85)
    print('')



# plot the shattering dimensionality
ax = line_plot_timevar(gs=gs[1, 3], fig=fig, x=x_data_time, y=shattering_dim_time[None, :, :], color=['grey', 'black'],
                          xlabel='time (s)', ylabel='accuracy', title='neural\ndimensionality', ylim=[0.45, 0.75],
                            xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                            patch_pars=patch_pars[2],
                            xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
print('******* Dimensionality decoding analysis *******')
plot_significance_stars(ax, sd_decoding[None,:], sd_decoding_rnd[:, None, :],
                        time_window=tws[2], ylim=[0.45, 0.75], tail='greater', y_position=0.7)
print('')



ax = line_plot_timevar(gs=gs[1, 2], fig=fig, x=x_data_time,
                       y=decoding_time[2, :, :][None, :, :], color=['grey', 'black'],
                       xlabel='time (s)', ylabel='accuracy', title=titles[2], ylim=y_lims,
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                       patch_pars=patch_pars[2],
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
print('******* Width decoding analysis *******')
plot_significance_stars(ax, decoding_list[2], decoding_list_rnd[2],
                        time_window=tws[2], ylim=[0.45, 0.8], tail='greater', y_position=0.85)
print('')


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
    plot_scatter(gs=gs[2, i_panel], fig=fig, x=x_data_r2[i_panel], y=y_data_r2[i_panel], scale=.2,
                 xaxis_label=xaxis_label_lis_r2[i_panel], yaxis_label=yaxis_label_lis_r2[i_panel],
                 overlay_model='contour', kde_smooth=1.2, out_factor=1.25)
    # if i_panel is odd
    if i_panel % 2 == 1:
        plot_cov(gs=gs[3, i_panel], fig=fig, dat=cov_dat[i_panel], scale=0.007)

    if i_panel % 2 == 0:
        plot_scatter(gs=gs[3, i_panel], fig=fig, x=x_data_r3[i_panel], y=y_data_r3[i_panel], scale=.2,
                     xaxis_label=xaxis_label_lis_r3[i_panel], yaxis_label=yaxis_label_lis_r3[i_panel],
                     overlay_model='contour', kde_smooth=1.2, out_factor=1.25)

print('******* Distance from random analysis *******')
plot_distance(gs[4, :2], fig, rnd_matrix, 'distance from random', pvals=p_vals_rnd)
print('')
print('******* Distance from minimal analysis *******')
plot_distance(gs[4, 2:], fig, str_matrix, 'distance from structured', pvals=p_vals_str)
print('')
plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'fig_2.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'fig_2.png', dpi=300)
print('')
print('Saved to ' + config['PATHS']['out_template_figures'])


