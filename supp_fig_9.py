from mpl_toolkits.axes_grid1 import make_axes_locatable

from fun_lib import *
with open('config.yml') as file:
    config = yaml.full_load(file)

print ('Generating Supplementary Figure 9... ')
print ('')


# %%
n_neurons = 100
max_noise = 2
n_trials = 200
n_bootstraps = 100

corrs_min, r2_min, corrs_rnd, r2_rnd, coefs_min_lis, coefs_rnd_lis = reg_par_recovery(n_neurons=n_neurons,
                                                                                      max_noise=max_noise,
                                                                                      n_trials=n_trials,
                                                                                      fit_to_noise=False,
                                                                                      n_bootstraps=n_bootstraps)

coefs_min = coefs_min_lis[-1]
coefs_rnd = coefs_rnd_lis[-1]

noise_levels = np.linspace(0.0001, max_noise, 9)

m_rnd = np.diag(np.cov(coefs_rnd)).mean()
n_circles = 4
rnd_std = m_rnd ** 0.5
r_all = rnd_std * 3
r_range = np.linspace(0.0, r_all, n_circles)

#ticks = [0, 1, 2, 3, 4]
ticks = [0, 1, 2]

# %%

size = 12
fig = plt.figure(figsize=(size * .66, size * 0.399))
gs = gridspec.GridSpec(3, 4, width_ratios=[.5, .5, .5, .5], figure=fig)
offset_axis = 26
sns.set_context('paper', rc={"lines.linewidth": 2})
err_style = 'bars'
n_figures = 24
dot_size = 10

col_rnd = sns.diverging_palette(230, 20, n=4)[-1]
col_opt = sns.diverging_palette(230, 20, n=4)[0]
unit_col_01 = 'dimgrey'
unit_col = 'darkgrey'
edge_color_01 = 'black'
edge_color = 'dimgrey'
scale = 4
ylim_r = [0.35, 1]
ylim_r2 = [-2.4, 1]
ylim_r2_ticks = [-2, 0, 1]


str_lim = scale - 1
y = np.linspace(-str_lim, str_lim, 40)
x = np.zeros(len(y))

ax = fig.add_subplot(gs[0])
m = np.mean(corrs_min, axis=-1)
s = np.std(corrs_min, axis=-1)
ax.plot(noise_levels, m[0, :], color=col_opt)
ax.fill_between(noise_levels, m[0, :] - s[0, :], m[0, :] + s[0, :], color=[col_opt], alpha=0.5)
ax.plot(noise_levels, m[1, :], color=col_rnd)
ax.fill_between(noise_levels, m[1, :] - s[1, :], m[1, :] + s[1, :], color=[col_rnd], alpha=0.5)
plt.ylim(ylim_r)
plt.xlabel('noise ($\\sigma$)')
plt.ylabel('correlation ($r$)')
sns.despine(right=True, top=True)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)

ax = fig.add_subplot(gs[1])
cols_sim = ['grey', 'black']
m = np.mean(r2_min, axis=-1)
s = np.std(r2_min, axis=-1)
ax.plot(noise_levels, m[0, :], color=col_opt)
ax.fill_between(noise_levels, m[0, :] - s[0, :], m[0, :] + s[0, :], color=[col_opt], alpha=0.5)
ax.plot(noise_levels, m[1, :], color=col_rnd)
ax.fill_between(noise_levels, m[1, :] - s[1, :], m[1, :] + s[1, :], color=[col_rnd], alpha=0.5)
plt.ylim(ylim_r2)
plt.xlabel('noise ($\\sigma$)')
plt.ylabel('quality of fit ($R^2$)')
sns.despine(right=True, top=True)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
ax.set_yticks(ylim_r2_ticks)
ax.set_yticklabels(ylim_r2_ticks)


ax = fig.add_subplot(gs[2])
m = np.mean(corrs_rnd, axis=-1)
s = np.std(corrs_rnd, axis=-1)
ax.plot(noise_levels, m[0, :], color=col_opt)
ax.fill_between(noise_levels, m[0, :] - s[0, :], m[0, :] + s[0, :], color=[col_opt], alpha=0.5)
ax.plot(noise_levels, m[1, :], color=col_rnd)
ax.fill_between(noise_levels, m[1, :] - s[1, :], m[1, :] + s[1, :], color=[col_rnd], alpha=0.5)
plt.ylim(ylim_r)
plt.xlabel('noise ($\\sigma$)')
plt.ylabel('correlation ($r$)')
sns.despine(right=True, top=True)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)

ax = fig.add_subplot(gs[3])
cols_sim = ['grey', 'black']
m = np.mean(r2_rnd, axis=-1)
s = np.std(r2_rnd, axis=-1)
ax.plot(noise_levels, m[0, :], color=col_opt)
ax.fill_between(noise_levels, m[0, :] - s[0, :], m[0, :] + s[0, :], color=[col_opt], alpha=0.5)
ax.plot(noise_levels, m[1, :], color=col_rnd)
ax.fill_between(noise_levels, m[1, :] - s[1, :], m[1, :] + s[1, :], color=[col_rnd], alpha=0.5)
plt.ylim(ylim_r2)
plt.xlabel('noise ($\\sigma$)')
plt.ylabel('quality of fit ($R^2$)')
sns.despine(right=True, top=True)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
ax.set_yticks(ylim_r2_ticks)
ax.set_yticklabels(ylim_r2_ticks)


ax = fig.add_subplot(gs[4])
ax.scatter(coefs_min[:, 0], coefs_min[:, 1], color=unit_col, s=dot_size, zorder=-5,
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
ax.set_xlabel('colour selectivity', labelpad=offset_axis)
ax.set_ylabel('shape selectivity', labelpad=offset_axis)
ax.scatter(x=[0.0], y=[0.0], color=col_opt, s=dot_size, zorder=5)

ax = fig.add_subplot(gs[5])
ax.scatter(coefs_min[:, 0, ], coefs_min[:, 2, ], color=unit_col, s=dot_size, zorder=-5,
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
ax.set_xlabel('colour selectivity', labelpad=offset_axis)
ax.set_ylabel('xor selectivity', labelpad=offset_axis)
ax.plot(x, y, color=col_opt, linewidth=3, zorder=5)

ax = fig.add_subplot(gs[8])
ax.scatter(coefs_min[:, 1], coefs_min[:, 2], color=unit_col, s=dot_size, zorder=-5,
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
ax.set_xlabel('shape selectivity', labelpad=offset_axis)
ax.set_ylabel('xor selectivity', labelpad=offset_axis)
ax.plot(x, y, color=col_opt, linewidth=3, zorder=5)

ax = fig.add_subplot(gs[9])
norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(np.cov(coefs_min.T), cmap=sns.diverging_palette(230, 20, as_cmap=True), norm=norm,
               origin='lower')
ax.set_yticks([0, 1, 2])
ax.set_xticks([0, 1, 2])
ax.set_yticklabels(['c', 's', 'xor'])
ax.set_xticklabels(['c', 's', 'xor'])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.formatter.set_powerlimits((0, 0))
ax.set_title('covariance\n(between coefficients)')

ax = fig.add_subplot(gs[6])
ax.scatter(coefs_rnd[:, 0], coefs_rnd[:, 1], color=unit_col, s=dot_size, zorder=-5,
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
ax.set_xlabel('colour selectivity', labelpad=offset_axis)
ax.set_ylabel('shape selectivity', labelpad=offset_axis)

for c in range(n_circles):
    circle = plt.Circle((0, 0), r_range[c], color=col_rnd, fill=False, zorder=-4)
    ax.add_patch(circle)

ax = fig.add_subplot(gs[7])
ax.scatter(coefs_rnd[:, 0, ], coefs_rnd[:, 2, ], color=unit_col, s=dot_size, zorder=-5,
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
ax.set_xlabel('colour selectivity', labelpad=offset_axis)
ax.set_ylabel('xor selectivity', labelpad=offset_axis)

for c in range(n_circles):
    circle = plt.Circle((0, 0), r_range[c], color=col_rnd, fill=False, zorder=-4)
    ax.add_patch(circle)

ax = fig.add_subplot(gs[10])
ax.scatter(coefs_rnd[:, 1], coefs_rnd[:, 2], color=unit_col, s=dot_size, zorder=-5,
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
ax.set_xlabel('shape selectivity', labelpad=offset_axis)
ax.set_ylabel('xor selectivity', labelpad=offset_axis)

for c in range(n_circles):
    circle = plt.Circle((0, 0), r_range[c], color=col_rnd, fill=False, zorder=-4)
    ax.add_patch(circle)

ax = fig.add_subplot(gs[11])
norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(np.cov(coefs_rnd.T), cmap=sns.diverging_palette(230, 20, as_cmap=True), norm=norm,
               origin='lower')
ax.set_yticks([0, 1, 2])
ax.set_xticks([0, 1, 2])
ax.set_yticklabels(['c', 's', 'xor'])
ax.set_xticklabels(['c', 's', 'xor'])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('covariance\n(between coefficients)')

fig.tight_layout()
plt.savefig(config['PATHS']['out_template_figures'] + 'supp_fig_9.svg')
plt.savefig( config['PATHS']['out_template_figures'] + 'supp_fig_9.png', dpi=300)
print('')
print('Saved to ' + config['PATHS']['out_template_figures'])