from typing import Dict, Any

from fun_lib import *
with open('config.yml') as file:
    config = yaml.full_load(file)


n_neurons = 100
n_coeffs = 3
opt_cov = np.diag([0, 0, 3])
seed_nb = 0
np.random.seed(seed_nb)
s_opt = np.random.multivariate_normal(np.zeros(n_coeffs),
                                      opt_cov,
                                      n_neurons)
np.random.seed(seed_nb)
rnd_cov = np.eye(3)
s_rnd = np.random.multivariate_normal(np.zeros(n_coeffs),
                                      rnd_cov,
                                      n_neurons)

conds = np.ones((4, 3))
conds[0, :2] = -1
conds[1, [0, 2]] = -1
conds[2, 1:] = -1
labels = [1, -1, -1, 1]  # XOR

alpha = [1, 0.66, 0.33, 0]
sel = []
titles = []
sig_sel = 0.7
for i_epoch in range(4):
    cov_comb = (rnd_cov * alpha[i_epoch]) + (opt_cov * (1 - alpha[i_epoch]))
    sel_comb = s_rnd = np.random.multivariate_normal(np.zeros(n_coeffs),
                                          cov_comb,
                                          n_neurons)
    sel_comb+= np.random.normal(0, sig_sel, (n_neurons, n_coeffs))
    sel.append(sel_comb)
    titles.append(str(round(alpha[i_epoch]*100)) + '% random')

print ('Generating Figure 1... ')
print ('')

for i in range(4):
    fig, gs = creat_plot_grid(1, 1, 4)
    ax = fig.add_subplot(gs[0], projection='3d')
    plot_state_space_3d(ax=ax, sel_matrix=sel[i], conds=conds, title=titles[i])
    plt.savefig(config['PATHS']['out_template_figures'] + 'fig_1_' + str(i+1) + '.svg')
    plt.savefig(config['PATHS']['out_template_figures'] + 'fig_1_' + str(i+1) + 'png', dpi=300)

fig, gs = creat_plot_grid(2, 4, 2)
x_data_r2 = [sel[0][:, config['ANALYSIS_PARAMS']['COLOUR_ID']], sel[0][:, config['ANALYSIS_PARAMS']['COLOUR_ID']],
             sel[-1][:, config['ANALYSIS_PARAMS']['COLOUR_ID']], sel[-1][:, config['ANALYSIS_PARAMS']['COLOUR_ID']]]
y_data_r2 = [sel[0][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], sel[0][:, -1],
             sel[-1][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], sel[-1][:, -1]]
x_data_r3 = [sel[0][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], None, sel[-1][:, config['ANALYSIS_PARAMS']['SHAPE_ID']], None]
y_data_r3 = [sel[0][:, -1], None, sel[-1][:, -1], None]


xaxis_label_lis_r2 = ['colour selectivity', 'colour selectivity', 'colour selectivity', 'colour selectivity']
yaxis_label_lis_r2 = ['shape selectivity', 'xor selectivity', 'shape selectivity', 'xor selectivity']
xaxis_label_lis_r3 = ['shape selectivity', None, 'shape selectivity', None]
yaxis_label_lis_r3 = ['xor selectivity', None, 'xor selectivity', None]
cov_dat = [None, sel[0], None, sel[-1]]

col_rnd = sns.diverging_palette(100, 20, n=4)[-1]
col_str = sns.diverging_palette(230, 20, n=20)[2]
col_str_edge = sns.diverging_palette(230, 20, n=20)[0]
col_r2 = [col_rnd, col_rnd, col_str, col_str]
col_r3 = [col_rnd, None, col_str, None]
edgecol_r2 = ['brown', 'brown', col_str_edge, col_str_edge]
edgecol_r3 = ['brown', None, col_str_edge, None]
for i_panel in range(4):
    plot_scatter(gs=gs[0, i_panel], fig=fig, x=x_data_r2[i_panel], y=y_data_r2[i_panel], scale=6,
                 xaxis_label=xaxis_label_lis_r2[i_panel], yaxis_label=yaxis_label_lis_r2[i_panel],
                 overlay_model='contour', dot_size=20, color=col_r2[i_panel], edgecolor=edgecol_r2[i_panel], offset_axis=30)
    # if i_panel is odd
    if i_panel % 2 == 1:
        plot_cov(gs=gs[1, i_panel], fig=fig, dat=cov_dat[i_panel], scale=3)

    if i_panel % 2 == 0:
        plot_scatter(gs=gs[1, i_panel], fig=fig, x=x_data_r3[i_panel], y=y_data_r3[i_panel], scale=6,
                     xaxis_label=xaxis_label_lis_r3[i_panel], yaxis_label=yaxis_label_lis_r3[i_panel],
                     overlay_model='contour', dot_size=20, color=col_r3[i_panel], edgecolor=edgecol_r3[i_panel],offset_axis=30)

plt.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'fig_1_5.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'fig_1_5.png', dpi=300)
print('')
print('Saved to ' + config['PATHS']['out_template_figures'])