from mpl_toolkits.axes_grid1 import make_axes_locatable

from fun_lib import *
with open('config.yml') as file:
    config = yaml.full_load(file)

print ('Generating Supplementary Figure 2... ')
print ('')


# %%
n_networks = 10
n_neurons = 400
n_coeffs = 3

low_reg_coeffs = np.zeros((n_neurons, n_coeffs, n_networks))
high_reg_coeffs = np.zeros((n_neurons, n_coeffs, n_networks))
for network in range(n_networks):
    name_low = config['PATHS']['output_path'] + 'optimized_selectivity_-1+1_lr0_005_sig20_reg0_400_2_' + str(network)
    name_high = config['PATHS']['output_path'] + 'optimized_selectivity_-1+1_lr0_005_sig20_reg20_400_2_' + str(network)

    low_reg_coeffs[:, :, network] = grab_variables(name_low)[0].T
    high_reg_coeffs[:, :, network] = grab_variables(name_high)[0].T


#%%

dec_and_cost = np.load(config['PATHS']['output_path'] + 'decodingxor_and_metabolic_cost_sig0w_w_b_reg_0_20_200_N400_sig2.npz')
xor_decoding_scores = dec_and_cost['xor_decoding_scores']
color_decoding_scores = dec_and_cost['color_decoding_scores']
shape_decoding_scores = dec_and_cost['shape_decoding_scores']
metabolic_costs = dec_and_cost['metabolic_costs']

# %%
n_bootstraps = 100
metric = 'euclidean distance'
normalisation = 'zscore'
rnd_model = 'gaussian (spherical)'
design_model = '+1/-1'
reg_levels = np.array([0, 1])

rnd_sel = np.zeros((n_networks, 3, len(reg_levels), n_bootstraps))
min_sel = np.zeros((n_networks, 3, len(reg_levels), n_bootstraps))
for network in range(n_networks):
    print('Computing distance from generative models for network ' + str(network))

    coeff_prepared = [
        np.concatenate([low_reg_coeffs[:, :, network][:, :, None], low_reg_coeffs[:, :, network][:, :, None]], axis=-1),
        np.concatenate([high_reg_coeffs[:, :, network][:, :, None], high_reg_coeffs[:, :, network][:, :, None]], axis=-1)
    ]
    _, _, rnd_matrix = dist_random(epochs=[coeff_prepared],
                                   n_bootstraps=n_bootstraps,
                                   rnd_model=rnd_model,
                                   model_names=['cue + shape'],
                                   bon_correction=False,
                                   metric=metric,
                                   design_model=design_model,
                                   relative_dist=True
                                   )

    _, _, str_matrix = dist_structured(epochs=[coeff_prepared],
                                       n_bootstraps=n_bootstraps,
                                       rnd_model=rnd_model,
                                       model_names=['cue + shape'],
                                       bon_correction=False,
                                       metric=metric,
                                       design_model=design_model,
                                       relative_dist=True
                                       )

    rnd_sel[network, :, :, :] = np.array(rnd_matrix)
    min_sel[network, :, :, :] = np.array(str_matrix)

# %% Generate random and optimal selectivity models looping over number of neurons

perf_units, N_list = xor_sim_units(n_neurons=401,
                                   step_neurons=10,
                                   noise_sig=5
                                   )


# %%

size = 12
fig = plt.figure(figsize=(size * .66, size * 0.532))
gs = gridspec.GridSpec(4, 4, width_ratios=[.5, .5, .5, .5], figure=fig)
offset_axis = 26
sns.set_context('paper', rc={"lines.linewidth": 2})
err_style = 'bars'
n_figures = 24
dot_size = 10
dist_dot = 30

col_rnd = sns.diverging_palette(230, 20, n=4)[-1]
col_opt = sns.diverging_palette(230, 20, n=4)[0]
col_data = colors.to_rgb('black')
col_data_edge = 'dimgrey'
unit_col = 'darkgrey'
edge_color = 'dimgrey'
scale = 4

str_lim = scale - 1
y = np.linspace(-str_lim, str_lim, 40)
x = np.zeros(len(y))

m_rnd = np.diag(np.cov(low_reg_coeffs[:, :, 0].T)).mean()
n_circles = 4
rnd_std = m_rnd ** 0.5
r_all = rnd_std * 3
r_range = np.linspace(0.0, r_all, n_circles)



ax = fig.add_subplot(gs[2, 0])
for network in range(n_networks):
    jitter = np.array([np.random.normal(0, 0.15),np.random.normal(0, 0.15)])
    ax.scatter(reg_levels + jitter, rnd_sel[network,0,:,:].mean(-1), color=col_data_edge, edgecolor=col_data, s=dist_dot)
plt.margins(x=0.1)

rnd_null = rnd_sel[:,1,:,:].flatten()
dec_mean = rnd_null.mean()
dec_std = rnd_null.std()
ax.axhline(dec_mean + 2 * dec_std, linestyle='--', linewidth=1, color=col_rnd)
ax.axhline(dec_mean - 2 * dec_std, linestyle='--', linewidth=1, color=col_rnd)
rect = patches.Rectangle((-1, dec_mean - 2 * dec_std), 3, 4 * dec_std, facecolor=col_rnd, alpha=0.5,
                         zorder=-8)
ax.add_patch(rect)

rnd_null = rnd_sel[:,2,:,:].flatten()
dec_mean = rnd_null.mean()
dec_std = rnd_null.std()
ax.axhline(dec_mean + 2 * dec_std, linestyle='--', linewidth=1, color=col_opt)
ax.axhline(dec_mean - 2 * dec_std, linestyle='--', linewidth=1, color=col_opt)
rect = patches.Rectangle((-1, dec_mean - 2 * dec_std), 3, 4 * dec_std, facecolor=col_opt, alpha=0.5,
                         zorder=-8)
ax.add_patch(rect)

ax.set_xticks([0, 1])
ax.set_xticklabels(['no reg.', 'high reg.'])
#ax.set_xlabel('regularisation')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
sns.despine(right=True, top=True)
ax.set_ylabel('relative\neuclidean dist.')
ax.set_title('relative distance\nfrom random ')


ax = fig.add_subplot(gs[2, 1])
for network in range(n_networks):
    jitter = np.array([np.random.normal(0, 0.15),np.random.normal(0, 0.15)])
    ax.scatter(reg_levels + jitter, min_sel[network,0,:,:].mean(-1), color=col_data_edge, edgecolor=col_data, s=dist_dot)

ax.set_xticks([0, 1])
ax.set_xticklabels(['no reg.', 'high reg.'])
#ax.set_xlabel('regularisation')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
sns.despine(right=True, top=True)
ax.set_ylabel('relative\neuclidean dist.')
ax.set_title('relative distance\nto minimal ')
plt.margins(x=0.1)

rnd_null = min_sel[:,1,:,:].flatten()
dec_mean = rnd_null.mean()
dec_std = rnd_null.std()
ax.axhline(dec_mean + 2 * dec_std, linestyle='--', linewidth=1, color=col_rnd)
ax.axhline(dec_mean - 2 * dec_std, linestyle='--', linewidth=1, color=col_rnd)
rect = patches.Rectangle((-1, dec_mean - 2 * dec_std), 3, 4 * dec_std, facecolor=col_rnd, alpha=0.5,
                         zorder=-8)
ax.add_patch(rect)

rnd_null = min_sel[:,2,:,:].flatten()
dec_mean = rnd_null.mean()
dec_std = rnd_null.std()
ax.axhline(dec_mean + 2 * dec_std, linestyle='--', linewidth=1, color=col_opt)
ax.axhline(dec_mean - 2 * dec_std, linestyle='--', linewidth=1, color=col_opt)
rect = patches.Rectangle((-1, dec_mean - 2 * dec_std), 3, 4 * dec_std, facecolor=col_opt, alpha=0.5,
                         zorder=-8)
ax.add_patch(rect)

ax = fig.add_subplot(gs[2, 2])
plt.margins(x=0.1)
for network in range(n_networks):
    jitter = np.array([np.random.normal(0, 0.15),np.random.normal(0, 0.15)])
    ax.scatter(reg_levels + jitter, xor_decoding_scores[network,:2].T, color=col_data_edge, edgecolor=col_data, s=dist_dot)

ax.set_xticks([0, 1])
ax.set_xticklabels(['no reg.', 'high reg.'])
#ax.set_xlabel('regularisation')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
sns.despine(right=True, top=True)
ax.set_ylabel('accuracy')
ax.set_title('xor decoding ')
ax.set_ylim([0.45, 1.1])
ax.axhline(0.5, linestyle='--', linewidth=1, color='grey')

ax = fig.add_subplot(gs[2, 3])
plt.margins(x=0.1)
for network in range(n_networks):
    jitter = np.array([np.random.normal(0, 0.15),np.random.normal(0, 0.15)])
    ax.scatter(reg_levels + jitter, metabolic_costs[network,:2].T, color=col_data_edge, edgecolor=col_data, s=dist_dot)

ax.set_xticks([0, 1])
ax.set_xticklabels(['no reg.', 'high reg.'])
#ax.set_xlabel('regularisation')
#plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
sns.despine(right=True, top=True)
ax.set_ylabel('$E (||\\mathbf{x}||^2) / N$')
ax.set_title('metabolic cost')
ax.set_ylim([-0.5, 4])


ax = fig.add_subplot(gs[0])
ax.scatter(low_reg_coeffs[:, :, 0][:, 0], low_reg_coeffs[:, :, 0][:, 1], color=unit_col, s=dot_size, zorder=-5,
           edgecolor=edge_color)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([-scale, scale])
ax.set_xlim([-scale, scale])
ax.set_yticks([-scale, scale])
ax.set_xticks([-scale, scale])
ax.set_aspect('equal')
ax.set_xlabel('colour', labelpad=offset_axis)
ax.set_ylabel('shape', labelpad=offset_axis)

for c in range(n_circles):
    circle = plt.Circle((0, 0), r_range[c], color=col_rnd, fill=False, zorder=-4)
    ax.add_patch(circle)

ax = fig.add_subplot(gs[1])
ax.scatter(low_reg_coeffs[:, :, 0][:, 0, ], low_reg_coeffs[:, :, 0][:, 2, ], color=unit_col, s=dot_size, zorder=-5,
           edgecolor=edge_color)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([-scale, scale])
ax.set_xlim([-scale, scale])
ax.set_yticks([-scale, scale])
ax.set_xticks([-scale, scale])
ax.set_aspect('equal')
ax.set_xlabel('colour', labelpad=offset_axis)
ax.set_ylabel('xor', labelpad=offset_axis)


for c in range(n_circles):
    circle = plt.Circle((0, 0), r_range[c], color=col_rnd, fill=False, zorder=-4)
    ax.add_patch(circle)


ax = fig.add_subplot(gs[4])
ax.scatter(low_reg_coeffs[:, :, 0][:, 1], low_reg_coeffs[:, :, 0][:, 2], color=unit_col, s=dot_size, zorder=-5,
           edgecolor=edge_color)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([-scale, scale])
ax.set_xlim([-scale, scale])
ax.set_yticks([-scale, scale])
ax.set_xticks([-scale, scale])
ax.set_aspect('equal')
ax.set_xlabel('shape', labelpad=offset_axis)
ax.set_ylabel('xor', labelpad=offset_axis)

for c in range(n_circles):
    circle = plt.Circle((0, 0), r_range[c], color=col_rnd, fill=False, zorder=-4)
    ax.add_patch(circle)


ax = fig.add_subplot(gs[5])
norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
im = ax.imshow(np.cov(low_reg_coeffs[:, :, 0].T), cmap=sns.diverging_palette(230, 20, as_cmap=True), norm=norm,
               origin='lower')
ax.set_yticks([0, 1, 2])
ax.set_xticks([0, 1, 2])
ax.set_yticklabels(['c', 's', 'xor'])
ax.set_xticklabels(['c', 's', 'xor'])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.formatter.set_powerlimits((0, 0))
ax.set_title('covariance')

ax = fig.add_subplot(gs[2])
ax.scatter(high_reg_coeffs[:, :, 0][:, 0], high_reg_coeffs[:, :, 0][:, 1], color=unit_col, s=dot_size, zorder=-5,
           edgecolor=edge_color)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([-scale, scale])
ax.set_xlim([-scale, scale])
ax.set_yticks([-scale, scale])
ax.set_xticks([-scale, scale])
ax.set_aspect('equal')
ax.set_xlabel('colour', labelpad=offset_axis)
ax.set_ylabel('shape', labelpad=offset_axis)
ax.scatter(x=[0.0], y=[0.0], color=col_opt, s=dot_size, zorder=5)

ax = fig.add_subplot(gs[3])
ax.scatter(high_reg_coeffs[:, :, 0][:, 0, ], high_reg_coeffs[:, :, 0][:, 2, ], color=unit_col, s=dot_size, zorder=-5,
           edgecolor=edge_color)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([-scale, scale])
ax.set_xlim([-scale, scale])
ax.set_yticks([-scale, scale])
ax.set_xticks([-scale, scale])
ax.set_aspect('equal')
ax.set_xlabel('colour', labelpad=offset_axis)
ax.set_ylabel('xor', labelpad=offset_axis)
ax.plot(x, y, color=col_opt, linewidth=2, zorder=5)

ax = fig.add_subplot(gs[6])
ax.scatter(high_reg_coeffs[:, :, 0][:, 1], high_reg_coeffs[:, :, 0][:, 2], color=unit_col, s=dot_size, zorder=-5,
           edgecolor=edge_color)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([-scale, scale])
ax.set_xlim([-scale, scale])
ax.set_yticks([-scale, scale])
ax.set_xticks([-scale, scale])
ax.set_aspect('equal')
ax.set_xlabel('shape', labelpad=offset_axis)
ax.set_ylabel('xor', labelpad=offset_axis)
ax.plot(x, y, color=col_opt, linewidth=2, zorder=5)

ax = fig.add_subplot(gs[7])
norm = colors.TwoSlopeNorm(vmin=-.2, vcenter=0, vmax=.2)
im = ax.imshow(np.cov(high_reg_coeffs[:, :, 0].T), cmap=sns.diverging_palette(230, 20, as_cmap=True), norm=norm,
               origin='lower')
ax.set_yticks([0, 1, 2])
ax.set_xticks([0, 1, 2])
ax.set_yticklabels(['c', 's', 'xor'])
ax.set_xticklabels(['c', 's', 'xor'])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('covariance')

ax = fig.add_subplot(gs[3, :2])

cols_sim = [
    col_rnd,
    col_opt,
]

m = np.mean(perf_units, axis=1)
std = np.std(perf_units, axis=1)
ax.plot(N_list, m[:, 0], color=cols_sim[0])
ax.plot(N_list, m[:, 1], color=cols_sim[1])
[plt.fill_between(N_list, m[:, i] - std[:, i], m[:, i] + std[:, i], color=cols_sim[i], alpha=0.5) for i in
 range(2)]
# ax.text(.75, .6, 'random mixed', color=cols_sim[0], zorder=5, ha='center')
plt.yticks([0.5, 0.75, 1])
plt.xlabel('noise (σ)')
plt.ylabel('XOR decoding')
ax.set_title('performance comparison')
ax.axhline(0.5, linewidth=1, linestyle='--', c='black')
sns.despine(right=True, top=True)

ax = fig.add_subplot(gs[3, 2])
plt.margins(x=0.1)
for network in range(n_networks):
    jitter = np.array([np.random.normal(0, 0.15),np.random.normal(0, 0.15)])
    ax.scatter(reg_levels + jitter, color_decoding_scores[network,:2].T, color=col_data_edge, edgecolor=col_data, s=dist_dot)

ax.set_xticks([0, 1])
ax.set_xticklabels(['no reg.', 'high reg.'])
#ax.set_xlabel('regularisation')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
sns.despine(right=True, top=True)
ax.set_ylabel('accuracy')
ax.set_title('colour decoding')
ax.set_ylim([0.40, 1.1])
ax.axhline(0.5, linestyle='--', linewidth=1, color='grey')

ax = fig.add_subplot(gs[3, 3])
plt.margins(x=0.1)
for network in range(n_networks):
    jitter = np.array([np.random.normal(0, 0.15),np.random.normal(0, 0.15)])
    ax.scatter(reg_levels + jitter, shape_decoding_scores[network,:2].T, color=col_data_edge, edgecolor=col_data, s=dist_dot)

ax.set_xticks([0, 1])
ax.set_xticklabels(['no reg.', 'high reg.'])
#ax.set_xlabel('regularisation')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
sns.despine(right=True, top=True)
ax.set_ylabel('accuracy')
ax.set_title('shape decoding')
ax.set_ylim([0.40, 1.1])
ax.axhline(0.5, linestyle='--', linewidth=1, color='grey')

fig.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'supp_fig_2.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'supp_fig_2.png', dpi=300)
print('')
print('Saved to ' + config['PATHS']['out_template_figures'])
