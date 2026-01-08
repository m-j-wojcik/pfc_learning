from fun_lib import *

with open('config.yml') as file:
    config = yaml.full_load(file)

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'run':
    data, labels = get_data_stages(observe_or_run='observe', file_name='exp2_f_rates_ses_time')

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)

    dec_time_res = run_time_resolved_dec_exp2(data, labels, if_save=True,
                                              fname='exp2_dec_timeres_xor2_'+ str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages',
                                              mode='within_stage')

if config['ANALYSIS_PARAMS']['OBSERVE_OR_RUN'] == 'observe':
    dec_exp2 = load_data(config['PATHS']['output_path'] + 'exp2_dec_timeres_xor2_'+ str(config['ANALYSIS_PARAMS']['N_STAGES']) +'stages' + '_within_stage.pickle')
    dec_time = dec_exp2['dec_time']

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

#%%
# create a grid for figure 3
fig, gs = creat_plot_grid(1, 4, 2)
x_data_time = np.linspace(-0.2, 1.5, dec_time.shape[0])

tw = config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK']
patch_pars = {'xy': ((tw[0] - 50) / 100, 0.0),
     'width': (tw[1] - tw[0]) / 100,
     'height': 1}
titles = ['context', 'task set', 'shape', 'width', 'xor']
# plot time-resolved decoding accuracy
print("Shape decoding")
ax = line_plot_timevar(gs=gs[0, 0], fig=fig, x=x_data_time, y=dec_time[:, 2, 0, :].T[None, :, :],
                       color=['grey', 'black'],
                       xlabel='time (s)', ylabel='accuracy', title=titles[2] + ' decoding', ylim=[0.45, 0.85],
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                       patch_pars=patch_pars,
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
plot_significance_stars(ax, scores_shape_late, scores_shape_rnd_late,
                       time_window=tw, ylim=[0.45, 0.85], tail='two')
print("XOR decoding")
ax = line_plot_timevar(gs=gs[0, 1], fig=fig, x=x_data_time, y=dec_time[:, 4, 0, :].T[None, :, :],
                       color=['grey', 'black'],
                       xlabel='time (s)', ylabel='accuracy', title=titles[4] + ' decoding', ylim=[0.45, 0.85],
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                       patch_pars=patch_pars,
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
plot_significance_stars(ax, scores_xor_late, scores_xor_rnd_late,
                       time_window=tw, ylim=[0.45, 0.85], tail='two')
print("Width decoding")
ax = line_plot_timevar(gs=gs[0, 2], fig=fig, x=x_data_time, y=dec_time[:, 3, 0, :].T[None, :, :],
                       color=['grey', 'black'],
                       xlabel='time (s)', ylabel='accuracy', title=titles[3] + ' decoding', ylim=[0.45, 0.85],
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                       patch_pars=patch_pars,
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
plot_significance_stars(ax, scores_width_late, scores_width_rnd_late,
                        time_window=tw, ylim=[0.45, 0.85], tail='two', y_position=0.79)
print("Width cross-set gen. decoding")
ax = line_plot_timevar(gs=gs[0, 3], fig=fig, x=x_data_time, y=dec_time[:, 3, 1, :].T[None, :, :],
                       color=['grey', 'black'],
                       xlabel='time (s)', ylabel='accuracy', title=titles[3] + ' cross-set gen.',
                       ylim=[0.35, 0.75],
                       xticks=[0.0, 0.5, 1.0], xticklabels=[0.0, 0.5, 1.0], baseline_line=0.5,
                       patch_pars=patch_pars,
                       xlim=[-0.2, 1.5], if_title=True, if_sem=None, xaxese_booo=True)
plot_significance_stars(ax, scores_width_late, scores_width_rnd_late,
                        time_window=tw, ylim=[0.45, 0.85], tail='two', dec_type=1, y_position=0.69)


# plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_supp_fig_exp2_dec_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.svg')
# plt.savefig(config['PATHS']['out_template_figures'] + 'rev1_supp_fig_exp2_dec_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]) + '_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]) + '.png', dpi=300)

plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'supp_fig_7_1.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'supp_fig_7_1.png', dpi=300)
#%%

fig, gs = creat_plot_grid(3, 4, 2, width_ratios=[0.5,0.63,.76,1])


stages_list = [3,4,5,6]
scale_scatter = 0.25
context_ylim = [0.42, 0.75]
xor_ylim = [0.45, 0.72]

y_lims = [context_ylim, xor_ylim, xor_ylim]

for i_stage, n_stage in enumerate(stages_list):
    dat_col_locked = load_data(config['PATHS']['output_path'] + 'exp2_decoding_collocked_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) + '_stages_' + str(n_stage) + '.pickle')
    dat_shape_locked = load_data(config['PATHS']['output_path'] + 'exp2_decoding_shapelocked_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(n_stage) + '.pickle')

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

    dat = load_data(config['PATHS']['output_path'] + 'exp2_decoding_within_stage_null_early_' +
                    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' +
                    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) +
                    '_late_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' +
                    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(n_stage) + '.pickle')
    scores_rnd_early = dat['scores_rnd_early']
    scores_rnd_late = dat['scores_rnd_late']

    print('*********** Distretising the learning trajectory: ' + str(n_stage) + ' stages ***********')
    print("Context coding")


    plot_dec_xgen(gs[0, i_stage], fig, scores_context_early, scores_context_rnd_early, title='context (' + str(n_stage) + ' stages)',
                  y_lim=context_ylim,
                  tails=['two', 'greater'], null_within=scores_rnd_early[1, :, :][None, :, :])
    print("Shape coding")
    plot_dec_xgen(gs[1, i_stage], fig, scores_shape_late, scores_shape_rnd_late, title='shape (' + str(n_stage) + ' stages)', y_lim=xor_ylim,
                  tails=['two', 'greater'], null_within=scores_rnd_late)
    print("XOR coding")
    plot_dec_xgen(gs[2, i_stage], fig, scores_xor_late, scores_xor_rnd_late, title='xor (' + str(n_stage) + ' stages)', y_lim=xor_ylim,
                  tails=['greater', 'greater'], null_within=scores_rnd_late)
plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'supp_fig_7_2.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'supp_fig_7_2.png', dpi=300)

fig1, gs1 = creat_plot_grid(3, 4, 2, width_ratios=[0.5,0.63,.76,1])
for i_stage, n_stage in enumerate(stages_list):
    dat_col_locked = load_data(config['PATHS']['output_path'] + 'exp2_decoding_collocked_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) + '_stages_' + str(n_stage) + '.pickle')
    dat_shape_locked = load_data(config['PATHS']['output_path'] + 'exp2_decoding_shapelocked_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' + str(
        config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(n_stage) + '.pickle')

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

    dat = load_data(config['PATHS']['output_path'] + 'exp2_decoding_within_stage_null_early_' +
                    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][0]) + '_' +
                    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_COL_LOCK'][1]) +
                    '_late_' + str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][0]) + '_' +
                    str(config['ANALYSIS_PARAMS']['TIME_WINDOW_EXP2_SHAPE_LOCK'][1]) + '_stages_' + str(n_stage) + '.pickle')
    scores_rnd_early = dat['scores_rnd_early']
    scores_rnd_late = dat['scores_rnd_late']

    mag_obs, mag_null_within, mag_null_ler = transform_xgen_dec_into_magnitude(
        decoding_matrix=scores_context_early,
        decoding_matrix_null=scores_rnd_early,
        decoding_matrix_null_ler=scores_context_rnd_early,
        chance=0.5
    )

    plot_magnitude(gs1[0, i_stage], fig1, mag_obs, mag_null_within, mag_null_ler=mag_null_ler,
                   title='context cross-set gen.')

    mag_obs, mag_null_within, mag_null_ler = transform_xgen_dec_into_magnitude(
        decoding_matrix=scores_shape_late,
        decoding_matrix_null=scores_rnd_late,
        decoding_matrix_null_ler=scores_shape_rnd_late,
        chance=0.5
    )
    print(mag_obs)

    plot_magnitude(gs1[1, i_stage], fig1, mag_obs, mag_null_within, mag_null_ler=mag_null_ler,
                   title='shape cross-set gen.')

    mag_obs, mag_null_within, mag_null_ler = transform_xgen_dec_into_magnitude(
        decoding_matrix=scores_xor_late,
        decoding_matrix_null=scores_rnd_late,
        decoding_matrix_null_ler=scores_xor_rnd_late,
        chance=0.5
    )

    plot_magnitude(gs1[2, i_stage], fig1, mag_obs, mag_null_within, mag_null_ler=mag_null_ler,
                   title='xor cross-set gen.')


plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'supp_fig_7_3.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'supp_fig_7_3.png', dpi=300)
print('')
print('Saved to ' + config['PATHS']['out_template_figures'])


