from mpl_toolkits.axes_grid1 import make_axes_locatable

from fun_lib import *
with open('config.yml') as file:
    config = yaml.full_load(file)

print ('Generating Supplementary Figure 3... ')
print ('')

# %% Generate random and optimal selectivity models looping over noise

perf, s_list = xor_sim_noise(sig_min=2.0,
                             sig_max=10.0,
                             sig_n=40,
                             n_neurons=1000,
                             )
# %% Fusi-style decoding analysis for random and optimal models
print ('')
print ('Running decoding and cross-generalised decoding...')
print ('')
m_decode, m_cross_gen = get_decoding_models(n_neurons=1000,
                                            noise_std=0.01,
                                            n_trials=20,
                                            seed=0,
                                            )

# %% generate selectivity models

n_coeffs = 3
n_neurons = 100
n_bootstraps = 1000
sig = 0
n_trials = 1
metric = 'euclidean'

distance_rnd = np.zeros((2, n_bootstraps))
distance_opt = np.zeros((2, n_bootstraps))

distance_from_rnd = np.zeros((4, n_bootstraps))
distance_to_opt = np.zeros((4, n_bootstraps))

for bootstrap in tqdm(range(n_bootstraps)):
    design = np.ones((4, n_coeffs))
    design[0, :2] = -1
    design[1, [0, 2]] = -1
    design[2, 1:] = -1

    opt_cov = np.diag([0, 0, 3])

    s_opt_1 = np.random.multivariate_normal(np.zeros(n_coeffs),
                                            opt_cov,
                                            n_neurons)

    s_opt_2 = np.random.multivariate_normal(np.zeros(n_coeffs),
                                            opt_cov,
                                            n_neurons)

    s_rnd_1 = np.random.multivariate_normal(np.zeros(n_coeffs),
                                            np.diag([1, 1, 1]),
                                            n_neurons)

    s_rnd_2 = np.random.multivariate_normal(np.zeros(n_coeffs),
                                            np.diag([1, 1, 1]),
                                            n_neurons)

    sel_0 = np.random.multivariate_normal(np.zeros(n_coeffs),
                                          np.diag([1, 1, 1]),
                                          n_neurons)

    sel_100 = np.random.multivariate_normal(np.zeros(n_coeffs),
                                            opt_cov,
                                            n_neurons)

    sel_33 = (sel_0 * 0.666) + (sel_100 * 0.333)
    sel_66 = (sel_0 * 0.333) + (sel_100 * 0.666)

    design_matrix = np.tile(design.T, n_trials).T

    sel_models = [s_opt_1, s_opt_2, s_rnd_1, s_rnd_2,
                  sel_33, sel_66]
    sels = []
    for _ in range(len(sel_models)):
        r_sel = sel_models[_] @ design_matrix.T
        r_sel = r_sel.T + sig * np.random.normal(0, 1, (4 * n_trials, n_neurons))
        r_sel = r_sel - np.mean(r_sel, axis=0, keepdims=True)
        sels.append(np.linalg.lstsq(design_matrix, r_sel)[0].T)

    if metric == 'euclidean':

        distance_rnd[0, bootstrap] = euclidean_distance(sels[2], sels[3])
        distance_rnd[1, bootstrap] = euclidean_distance(sels[2], sels[0])

        distance_opt[0, bootstrap] = euclidean_distance(sels[0], sels[1])
        distance_opt[1, bootstrap] = euclidean_distance(sels[0], sels[2])

        distance_to_opt[3, bootstrap] = euclidean_distance(sels[0], sels[1])
        distance_to_opt[2, bootstrap] = euclidean_distance(sels[0], sels[5])
        distance_to_opt[1, bootstrap] = euclidean_distance(sels[0], sels[4])
        distance_to_opt[0, bootstrap] = euclidean_distance(sels[0], sels[2])

        distance_from_rnd[0, bootstrap] = euclidean_distance(sels[2], sels[3])
        distance_from_rnd[1, bootstrap] = euclidean_distance(sels[2], sels[4])
        distance_from_rnd[2, bootstrap] = euclidean_distance(sels[2], sels[5])
        distance_from_rnd[3, bootstrap] = euclidean_distance(sels[2], sels[0])

    elif metric == 'KL':

        distance_rnd[0, bootstrap] = KLdivergence(sels[2], sels[3])
        distance_rnd[1, bootstrap] = KLdivergence(sels[2], sels[0])

        distance_opt[0, bootstrap] = KLdivergence(sels[0], sels[1])
        distance_opt[1, bootstrap] = KLdivergence(sels[0], sels[2])

        distance_to_opt[3, bootstrap] = KLdivergence(sels[0], sels[1])
        distance_to_opt[2, bootstrap] = KLdivergence(sels[0], sels[5])
        distance_to_opt[1, bootstrap] = KLdivergence(sels[0], sels[4])
        distance_to_opt[0, bootstrap] = KLdivergence(sels[0], sels[2])

        distance_from_rnd[0, bootstrap] = KLdivergence(sels[2], sels[3])
        distance_from_rnd[1, bootstrap] = KLdivergence(sels[2], sels[4])
        distance_from_rnd[2, bootstrap] = KLdivergence(sels[2], sels[5])
        distance_from_rnd[3, bootstrap] = KLdivergence(sels[2], sels[0])

distance_from_rnd = distance_from_rnd - distance_rnd[0, :].mean()
distance_to_opt = distance_to_opt - distance_opt[0, :].mean()

distance_rnd = distance_rnd - distance_rnd[0, :].mean()
distance_opt = distance_opt - distance_opt[0, :].mean()
distance_opt_cnst = np.tile(distance_opt[0, :][None, :].T, 4).T
distance_rnd_cnst = np.tile(distance_rnd[0, :][None, :].T, 4).T
distance_opt_cnst_pre = np.tile(distance_rnd[1, :][None, :].T, 4).T
distance_rnd_cnst_pre = np.tile(distance_opt[1, :][None, :].T, 4).T

distance_opt_cnst /= np.mean(distance_opt_cnst_pre, axis=-1, keepdims=True)
distance_to_opt /= np.mean(distance_opt_cnst_pre, axis=-1, keepdims=True)
distance_opt_cnst_pre /= np.mean(distance_opt_cnst_pre, axis=-1, keepdims=True)

distance_rnd_cnst /= np.mean(distance_rnd_cnst_pre, axis=-1, keepdims=True)
distance_from_rnd /= np.mean(distance_rnd_cnst_pre, axis=-1, keepdims=True)
distance_rnd_cnst_pre /= np.mean(distance_rnd_cnst_pre, axis=-1, keepdims=True)

# %% plot


size = 12
fig = plt.figure(figsize=(size * .66, size * 0.532))
gs = gridspec.GridSpec(4, 4, width_ratios=[.5, .5, .5, .5], figure=fig)
# gs = gridspec.GridSpec(4, 4, width_ratios=[.5, .5, .5, .5], figure=fig)
offset_axis = 26
sns.set_context('paper', rc={"lines.linewidth": 2})
err_style = 'bars'
n_figures = 24
dot_size = 10
dot_size2 = 20
frate_scale = 8
n_parts = 4
dist_dot = 30

col_blue = [[74 / 255, 148 / 255, 242 / 255]]
col_green = [[0 / 255, 168 / 255, 143 / 255]]

col_data = colors.to_rgb('black')
col_rnd = sns.diverging_palette(230, 20, n=2)[1]
col_opt = sns.diverging_palette(230, 20, n=2)[0]
seed_nb = 0

n_coeffs = 3
N_neurons = 100
opt_cov = np.diag([0, 0, 3])

np.random.seed(seed_nb)
s_opt = np.random.multivariate_normal(np.zeros(n_coeffs),
                                      opt_cov,
                                      N_neurons)
np.random.seed(seed_nb)
rnd_cov = np.eye(3)
s_rnd = np.random.multivariate_normal(np.zeros(n_coeffs),
                                      rnd_cov,
                                      N_neurons)

cols_sim = [
    col_rnd,
    col_opt,
]

ax = fig.add_subplot(gs[0])
ax.scatter(s_rnd[:, 0], s_rnd[:, 1], color=cols_sim[0], s=dot_size, zorder=-5)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_yticks([-6, 6])
ax.set_xticks([-6, 6])
ax.set_aspect('equal')
ax.set_xlabel('colour selectivity', labelpad=offset_axis)
ax.set_ylabel('shape selectivity', labelpad=offset_axis)

ax = fig.add_subplot(gs[1])
ax.scatter(s_rnd[:, 0], s_rnd[:, 2], color=cols_sim[0], s=dot_size, zorder=-5)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_yticks([-6, 6])
ax.set_xticks([-6, 6])
ax.set_aspect('equal')
ax.set_xlabel('colour selectivity', labelpad=offset_axis)
ax.set_ylabel('xor selectivity', labelpad=offset_axis)

ax = fig.add_subplot(gs[4])
ax.scatter(s_rnd[:, 1], s_rnd[:, 2], color=cols_sim[0], s=dot_size, zorder=-5)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_yticks([-6, 6])
ax.set_xticks([-6, 6])
ax.set_aspect('equal')
ax.set_xlabel('shape selectivity', labelpad=offset_axis)
ax.set_ylabel('xor selectivity', labelpad=offset_axis)

ax = fig.add_subplot(gs[2])
ax.scatter(s_opt[:, 0], s_opt[:, 1], color=cols_sim[1], s=dot_size, zorder=-5)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_yticks([-6, 6])
ax.set_xticks([-6, 6])
ax.set_aspect('equal')
ax.set_xlabel('colour selectivity', labelpad=offset_axis)
ax.set_ylabel('shape selectivity', labelpad=offset_axis)

ax = fig.add_subplot(gs[3])
ax.scatter(s_opt[:, 0], s_opt[:, 2], color=cols_sim[1], s=dot_size, zorder=-5)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_yticks([-6, 6])
ax.set_xticks([-6, 6])
ax.set_aspect('equal')
ax.set_xlabel('colour selectivity', labelpad=offset_axis)
ax.set_ylabel('xor selectivity', labelpad=offset_axis)

ax = fig.add_subplot(gs[6])
ax.scatter(s_opt[:, 1], s_opt[:, 2], color=cols_sim[1], s=dot_size, zorder=-5)
# move the left spine (y axis) to the right
ax.spines['left'].set_position(('axes', 0.5))
# move the bottom spine (x axis) up
ax.spines['bottom'].set_position(('axes', 0.5))
# turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_yticks([-6, 6])
ax.set_xticks([-6, 6])
ax.set_aspect('equal')
ax.set_xlabel('shape selectivity', labelpad=offset_axis)
ax.set_ylabel('xor selectivity', labelpad=offset_axis)

ax = fig.add_subplot(gs[5])
norm = colors.TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=3)
im = ax.imshow(np.diag([1, 1, 1]), cmap=sns.diverging_palette(230, 20, as_cmap=True), norm=norm, origin='lower')
ax.set_yticks([0, 1, 2])
ax.set_xticks([0, 1, 2])
ax.set_yticklabels(['c', 's', 'xor'])
ax.set_xticklabels(['c', 's', 'xor'])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('covariance \n (selectivity coefficients) ')

ax = fig.add_subplot(gs[7])
norm = colors.TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=3)
im = ax.imshow(np.diag([0, 0, 3]), cmap=sns.diverging_palette(230, 20, as_cmap=True), norm=norm, origin='lower')
ax.set_yticks([0, 1, 2])
ax.set_xticks([0, 1, 2])
ax.set_yticklabels(['c', 's', 'xor'])
ax.set_xticklabels(['c', 's', 'xor'])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('covariance \n (selectivity coefficients) ')

# plot distance from random

ax = fig.add_subplot(gs[2, :2])
ax.scatter(list(range(n_parts)), distance_from_rnd.mean(-1), color=col_data, s=dist_dot)
ax.scatter(list(range(n_parts)), distance_rnd_cnst.mean(-1), color=col_rnd, s=dist_dot)
ax.scatter(list(range(n_parts)), distance_opt_cnst_pre.mean(-1), color=col_opt, s=dist_dot)
ax.errorbar(list(range(n_parts)), distance_from_rnd.mean(-1), fmt='', yerr=distance_from_rnd.std(-1), color=col_data,
            alpha=0.5)
ax.errorbar(list(range(n_parts)), distance_rnd_cnst.mean(-1), fmt='', yerr=distance_rnd_cnst.std(-1), color=col_rnd,
            alpha=0.5)
ax.errorbar(list(range(n_parts)), distance_opt_cnst_pre.mean(-1), fmt='', yerr=distance_opt_cnst.std(-1), color=col_opt,
            alpha=0.5)
ax.set_xticks(list(range(n_parts)))
ax.set_xticklabels([0, 33, 66, 100])
ax.set_yticks([0, 1])
ax.set_xlabel('% of minimal selectivity')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
sns.despine(right=True, top=True)
ax.set_ylabel('relative\neuclidean dist.')
ax.set_title('relative distance from random ')

# plot distance to minimal

ax = fig.add_subplot(gs[3, :2])
ax.scatter(list(range(n_parts)), distance_to_opt.mean(-1), color=col_data, s=dist_dot)
ax.scatter(list(range(n_parts)), distance_rnd_cnst_pre.mean(-1), color=col_rnd, s=dist_dot)
ax.scatter(list(range(n_parts)), distance_opt_cnst.mean(-1), color=col_opt, s=dist_dot)
ax.errorbar(list(range(n_parts)), distance_to_opt.mean(-1), fmt='', yerr=distance_from_rnd.std(-1), color=col_data,
            alpha=0.5)
ax.errorbar(list(range(n_parts)), distance_rnd_cnst_pre.mean(-1), fmt='', yerr=distance_rnd_cnst.std(-1), color=col_rnd,
            alpha=0.5)
ax.errorbar(list(range(n_parts)), distance_opt_cnst.mean(-1), fmt='', yerr=distance_opt_cnst.std(-1), color=col_opt,
            alpha=0.5)
ax.set_xticks(list(range(n_parts)))
ax.set_xticklabels([0, 33, 66, 100])
ax.set_yticks([0, 1])
ax.set_xlabel('% of minimal selectivity')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
sns.despine(right=True, top=True)
ax.set_ylabel('relative\neuclidean dist.')
ax.set_title('relative distance to minimal ')

ax = fig.add_subplot(gs[3, 2])
ax.plot(m_decode[:, 0], color=col_rnd)
ax.scatter(['colour', 'shape', 'xor'], m_decode[:, 0], color=col_rnd, s=dot_size2)
ax.plot(m_decode[:, 1], color=col_opt)
ax.scatter(['colour', 'shape', 'xor'], m_decode[:, 1], color=col_opt, s=dot_size2)
ax.axhline(0.5, linestyle='--', linewidth=1, color='black', zorder=-1)
ax.set_ylabel('accuracy')
ax.set_title('decoding')
ax.set_xticks([0, 1, 2], ['colour', 'shape', 'xor'])
ax.set_ylim([0, 1.09])
# ax.legend(['random mixed sel.', 'optimal structured sel.'])
sns.despine(right=True, top=True)
plt.margins(x=0.1)

ax = fig.add_subplot(gs[3, 3])
ax.plot(m_cross_gen[:, 0], color=col_rnd)
ax.scatter(['colour', 'shape', 'xor'], m_cross_gen[:, 0], color=col_rnd, s=dot_size2)
ax.plot(m_cross_gen[:, 1], color=col_opt)
ax.scatter(['colour', 'shape', 'xor'], m_cross_gen[:, 1], color=col_opt, s=dot_size2)
ax.axhline(0.5, linestyle='--', linewidth=1, color='black', zorder=-1)
ax.set_ylabel('accuracy')
ax.set_title('cross-gen. \n decoding')
ax.set_xticks([0, 1, 2], ['colour', 'shape', 'xor'])
ax.set_ylim([0, 1.09])
# ax.legend(['random mixed sel.', 'optimal structured sel.'])
sns.despine(right=True, top=True)
plt.margins(x=0.1)

ax = fig.add_subplot(gs[2, 2:])
m = np.mean(perf, axis=1)
std = np.std(perf, axis=1)
ax.plot(s_list, m[:, 0], color=cols_sim[0])
ax.plot(s_list, m[:, 1], color=cols_sim[1])
[plt.fill_between(s_list, m[:, i] - std[:, i], m[:, i] + std[:, i], color=cols_sim[i], alpha=0.5, clip_on=False) for i in
 range(2)]
# ax.text(.75, .6, 'random mixed', color=cols_sim[0], zorder=5, ha='center')
plt.yticks([0.5, 0.75, 1])
plt.xlabel('noise (σ)')
plt.ylabel('accuracy')
ax.set_title('performance comparison (xor decoding)')
ax.axhline(0.5, linewidth=1, linestyle='--', c='black')
sns.despine(right=True, top=True)
ax.set_ylim([None, 1.01])
fig.tight_layout()

plt.savefig(config['PATHS']['out_template_figures'] + 'supp_fig_3.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'supp_fig_3.png', dpi=300)
print('')
print('Saved to ' + config['PATHS']['out_template_figures'])
