from fun_lib import *
with open('config.yml') as file:
    config = yaml.full_load(file)


if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN_PREPROCESSING'] == 'run':
    ses_all = combine_session_lists(mode='fix_bias', which_exp='exp1', combine_all=False)

    data, labels = get_data_stages(observe_or_run='run', session_list=ses_all,
                                   file_name='exp1_f_rates_ses_fix_bias_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages', return_data=True)

# compute the switch costs for the proportion of fixation breaks between rewarded and non-rewarded trials as a function of learning
if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Running behavioural analysis')
    if config['ANALYSIS_PARAMS']['N_STAGES'] < 5:
        ses_all = combine_session_lists(mode='stages', which_exp='exp1', combine_all=False)
    else:
        # when discretising learning into more than 4 stages use a sliding window approach to maintain sufficient statistical
        # power (have at least ~70 neurons per stage)
        ses_all = combine_session_lists(mode='sliding_window', which_exp='exp1', combine_all=False)
    switch_costs_all = get_switch_costs(ses_all)

    save_data([switch_costs_all], ['switch_costs_all'], config['PATHS']['output_path'] + 'exp1_switch_costs' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages' +'.pickle')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    swich_costs = load_data(config['PATHS']['output_path'] + 'exp1_switch_costs' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages' +'.pickle')['switch_costs_all']
    stages_colour_switch, stages_shape_switch, stages_hier_switch = swich_costs

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Run a time-resolved xgen decoding analysis')
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_time_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages')

    xgen_decoding = run_time_resolved_xgen(data, labels, if_save=True,
                                           fname='exp1_xgen_timeres_' + str(
                                               config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages',
                                           mode='within_stage')
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_xgen_timeres_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '_within_stage.pickle')
    xgen_decoding_time = data_exp1['xgen_time'].T

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_timeres_' + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '_within_stage.pickle')
    shattering_dim_time = data_exp1['shattering_dim_time'].mean(-1).T
    decoding_time = data_exp1['decoding_time'].T
else:
    raise ValueError('Processed data missing. Run analyses in figure_2.py')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Computing selectivity coefficients')
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias_' + str(
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
              config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_fixbias_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][1])  + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '.pickle')
    selectivity_coefficients = [selectivity_coefficients_xval[_].mean(-1) for _ in range(config['ANALYSIS_PARAMS']['N_STAGES'])]
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_fixbias_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][1]) + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '.pickle')
    selectivity_coefficients_xval = data_exp1['selectivity_coefficients_xval']
    selectivity_coefficients = [selectivity_coefficients_xval[_].mean(-1) for _ in range(config['ANALYSIS_PARAMS']['N_STAGES'])]



if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Computing similarity to structured/random models')
    data_exp1 = load_data(config['PATHS']['output_path'] + 'selectivity_coefficients_exp1_fixbias_' + str(
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
              config['PATHS']['output_path'] + 'distance_from_random_exp1_fixbias_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][1]) + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '.pickle')
    p_vals_rnd = p_vals_rnd[0]
    p_vals_str = p_vals_str[0]

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'distance_from_random_exp1_fixbias_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SEL'][1]) + str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages'+ '.pickle')
    rnd_matrix = data_exp1['rnd_matrix']
    p_vals_rnd = data_exp1['p_vals_rnd'][0]
    str_matrix = data_exp1['str_matrix']
    p_vals_str = data_exp1['p_vals_str'][0]

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':

    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_time_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages')
    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'], n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=98)

    pc_var = run_pca_movewin(data_eq, labels_eq, n_comps=4, factor=[0, 0, 1, 1, 2, 2, 3, 3])
    pc_var_ler_null = run_pca_movewin_null(data_eq[0], data_eq[-1], labels_eq[0], n_comps=4, factor=[0, 0, 1, 1, 2, 2, 3, 3])
    save_data([pc_var, pc_var_ler_null], ['pc_var', 'pc_var_ler_null'], config['PATHS']['output_path'] + 'exp1_supp_fig_pc_var_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')
elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_supp_fig_pc_var_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')
    pc_var = data_exp1['pc_var']
    pc_var_ler_null = data_exp1['pc_var_ler_null']


if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    late_data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_shapelocked_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')

    late_decoding = late_data_exp1['late_xgen_decoding']
    late_decoding_ler_rnd = late_data_exp1['late_xgen_decoding_ler_rnd']
else:
    raise ValueError('Processed data missing. Run analyses in figure_2.py')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    late_data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_fixbias_shapelocked_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')
    early_data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_fixbias_collocked_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')


    late_decoding = late_data_exp1['decoding']
    late_decoding_ler_rnd = late_data_exp1['decoding_rnd']

    late_shattering_dim = late_data_exp1['shattering_dim']
    late_shattering_dim_ler_rnd = late_data_exp1['shattering_dim_rnd']

    early_decoding = early_data_exp1['decoding']
    early_decoding_ler_rnd = early_data_exp1['decoding_rnd']

    decoding_list = [early_decoding[:, 0][None,:], late_decoding[:, 1][None,:], late_decoding[:, 2][None,:], late_decoding[:, 3][None,:]]
    decoding_list_rnd = [early_decoding_ler_rnd[:,:,0][:,None,:], late_decoding_ler_rnd[:,:,1][:,None,:], late_decoding_ler_rnd[:,:,2][:,None,:], late_decoding_ler_rnd[:,:,3][:,None,:]]
    sd_decoding = late_shattering_dim.mean(-1)
    sd_decoding_rnd = late_shattering_dim_ler_rnd.mean(-1)

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    late_data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_shapelocked_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_SHAPE_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')
    early_data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_collocked_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP1_COL_LOCK'][1]) + '_' + str(
        config['ANALYSIS_PARAMS']['N_STAGES']) + 'stages' + '.pickle')

    late_xgen_decoding = late_data_exp1['late_xgen_decoding']
    late_xgen_decoding_ler_rnd = late_data_exp1['late_xgen_decoding_ler_rnd']

    early_xgen_decoding = early_data_exp1['early_xgen_decoding']
    early_xgen_decoding_ler_rnd = early_data_exp1['early_xgen_decoding_ler_rnd']

    xgen_decoding_list = [late_xgen_decoding[:, 0][None,:], late_xgen_decoding[:, 1][None,:], late_xgen_decoding[:, 2][None,:], late_xgen_decoding[:, 3][None,:]]
    xgen_decoding_list_rnd = [late_xgen_decoding_ler_rnd[:,:,0][:,None,:], late_xgen_decoding_ler_rnd[:,:,1][:,None,:], late_xgen_decoding_ler_rnd[:,:,2][:,None,:], late_xgen_decoding_ler_rnd[:,:,3][:,None,:]]

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    print('Run a time-resolved decoding analysis')
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp1_f_rates_ses_fix_bias')

    shattering_dim_time, decoding_time, decoding_null_time = run_time_resolved_decoding(data, labels, if_save=True, fname='exp1_decoding_timeres_fix_bias', mode='within_stage')

elif config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    data_exp1 = load_data(config['PATHS']['output_path'] + 'exp1_decoding_timeres_fix_bias_within_stage.pickle')
    shattering_dim_time_fix = data_exp1['shattering_dim_time'].mean(-1).T
    decoding_time_fix = data_exp1['decoding_time'].T

#%%

print ('Generating Supplementary Figure 4... ')
print ('')

# create a grid for the decoding supp. figure
fig, gs = creat_plot_grid(3, 5, 2)


plot_mean_and_ci_prop(gs=gs[0], fig=fig, reward_prop=stages_shape_switch, n_perm=100000, title='shape SC', y_label= "stay vs switch", baseline_val=0.0, vmin=-1, vmax=4)
plot_mean_and_ci_prop(gs=gs[1], fig=fig, reward_prop=stages_colour_switch, n_perm=100000, title='colour SC', y_label= "stay vs switch", baseline_val=0.0, vmin=-1, vmax=4)
plot_mean_and_ci_prop(gs=gs[2], fig=fig, reward_prop=stages_hier_switch, n_perm=100000, title='hierarchy SC', y_label= "colour vs shape SC", baseline_val=0.0, vmin=-1, vmax=4)


x_data_time = np.linspace(-0.2, 1.5, shattering_dim_time.shape[-1])
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
titles = ['colour', 'shape', 'width', 'xor']
y_lims = [0.45, 0.95]

print('******* PCA analysis (stage 1 vs stage 4) *******')
plot_pca(gs[3], fig, pc_var, pc_var_ler_null, title='First PC', y_lim=[.40,.60])
print('')

tails = []
# plot time-resolved decoding accuracy
for i_panel in range(4):
    skip_width = 0
    if i_panel == 2:
        skip_width = 1
    ax = line_plot_timevar(gs=gs[i_panel+4], fig=fig, x=x_data_time, y=decoding_time_fix[i_panel, :, :][None, :, :], color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel], ylim=y_lims,
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars[i_panel],
                           xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
    print('******* ' + titles[i_panel] + ' decoding analysis (low vs high trial termination) *******')
    plot_significance_stars(ax, decoding_list[i_panel], decoding_list_rnd[i_panel],
                            time_window=tws[i_panel], ylim=[0.45, 0.8], tail='greater', y_position=0.85)
    print('')



# plot the shattering dimensionality
ax = line_plot_timevar(gs=gs[8], fig=fig, x=x_data_time, y=shattering_dim_time_fix[None, :, :], color=['grey', 'black'],
                          xlabel='time (s)', ylabel='accuracy', title='neural\ndimensionality', ylim=[0.45, 0.75],
                            xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                            patch_pars=patch_pars[2],
                            xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
print('******* Dimensionality decoding analysis (low vs high trial termination) *******')
plot_significance_stars(ax, sd_decoding[None,:], sd_decoding_rnd[:, None, :],
                        time_window=tws[2], ylim=[0.45, 0.75], tail='greater', y_position=0.7)
print('')

# ses_all = combine_session_lists(mode='fix_bias', which_exp='exp1')
# fix_breaks_animals = get_fixation_breaks([ses_all], experiment_label='exp1')[0]
# no_reward_fix = fix_breaks_animals[:, 2] / fix_breaks_animals[:, 3]
# reward_fix = fix_breaks_animals[:, 1] / fix_breaks_animals[:, 3]
# dat = no_reward_fix / reward_fix
# data_prop_s = np.array_split(dat, 4)
# mean_fix_prop = [str(round(np.nanmean(_), 1)) for _ in data_prop_s]
mean_fix_prop = ['1.1', '1.3', '1.9', '3.3']
print('******* Distance from random analysis (low vs high trial termination) *******')
plot_distance(gs[9], fig, rnd_matrix, 'distance from\nrandom', pvals=p_vals_rnd, x_stages_ticks=mean_fix_prop, x_label='trial termination bias')
print('')
print('******* Distance from minimal analysis (low vs high trial termination) *******')
plot_distance(gs[10], fig, str_matrix, 'distance from\nstructured', pvals=p_vals_str , x_stages_ticks=mean_fix_prop, x_label='trial termination bias')
print('')

tws=[tw_shape,tw_shape,tw_shape,tw_shape]
patch_pars = [
    {'xy': ((tw_shape[0] - 50) / 100, 0.0),
     'width': (tw_shape[1] - tw_shape[0]) / 100,
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


tails =['greater', 'greater', 'smaller']

# plot time-resolved decoding accuracy
for i_panel in range(3):
    skip_width = 0
    if i_panel == 2:
        skip_width = 1
    ax = line_plot_timevar(gs=gs[i_panel+ 11], fig=fig, x=x_data_time,
                           y=xgen_decoding_time[i_panel+skip_width, :, :][None, :, :], color=['grey', 'black'],
                           xlabel='time (s)', ylabel='accuracy', title=titles[i_panel+skip_width] + '\ncross. gen. dec.', ylim=[0.25, 0.95],
                           xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                           patch_pars=patch_pars[i_panel+skip_width],
                           xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
    print('******* ' + titles[i_panel+skip_width] + ' cross-generalisation analysis (stage 1 vs stage 4) *******')
    plot_significance_stars(ax, xgen_decoding_list[i_panel+skip_width], xgen_decoding_list_rnd[i_panel+skip_width],
                            time_window=tws[i_panel+skip_width], ylim=[0.45, 0.8], tail=tails[i_panel], y_position=0.85)
    print('')

plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'supp_fig_4.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'supp_fig_4.png', dpi=300)
print('')
print('Saved to ' + config['PATHS']['out_template_figures'])
