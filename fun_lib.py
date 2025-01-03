import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import spearmanr, stats
from tqdm import tqdm
from itertools import groupby
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import random
import itertools
from matplotlib import colors
from mne.decoding import SlidingEstimator, GeneralizingEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC as SVM
from scipy.ndimage import label
import pickle
from matplotlib import patches
from numpy.linalg import norm
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

plt.rcParams['svg.fonttype'] = 'none'
from itertools import chain
from mne.decoding import cross_val_multiscore
import scipy.io as io
import yaml



def assign_lables(labels, factor):
    conditions = np.unique(labels)
    res = dict(zip(conditions, factor))

    return list(map(res.get, labels))


def bias_corr(sel):
    conds = np.zeros((4, 3))  # conditions x selectivity (pure color, pure shape, interaction)
    conds[0, 0] = 1
    conds[1, 1] = 1  # Not rewarded
    conds[2, :] = 0
    conds[3, :] = 1  # rewarded

    const = np.array([1, 1, 1, 1])
    X = np.vstack([const, conds[:, 0], conds[:, 1], conds[:, 2]]).T

    rates = sel @ conds.T
    coeffs = sp.linalg.lstsq(X, rates.T)[0][1:, :]
    cov_new = np.cov(coeffs)
    new_sel = np.random.multivariate_normal(np.zeros(sel.shape[1]),
                                            cov_new,
                                            sel.shape[0])
    return new_sel


def downsample_data(*arg, fact=10):
    retval = []
    # downsample trailing dimension (i.e. time axis) by factor 10
    for dat in arg:
        shape = list(dat.shape)
        shape[-1] = shape[-1] // fact
        dat = dat.reshape(shape + [fact])
        retval.append(np.mean(dat, axis=len(shape)))
    return np.array(retval)[0, :, :, :]


def get_data(session_list, path_spikes, path_meta, window, cut_off=False):
    parts = session_list
    data_all_parts = []
    labels_all_parts = []

    for p in range(len(parts)):

        sessions = parts[p]

        print('Loading and combining the spiking data: part ', p + 1, '/', len(parts))
        units_dat = []
        units_labels = []
        for s in tqdm(range(len(sessions))):
            spks = np.load(path_spikes.format(sessions[s]))
            spks = spks[:, :, 500:3000]
            spks = downsample_data(spks)
            trials = np.load(path_meta.format(sessions[s]))
            labels = trials[trials != 0]
            spks_data = spks[trials != 0, :, window[0]:window[1]]
            spks_data = spks_data[:, :, :]
            if cut_off:
                spks_data = spks_data[:cut_off, :, :]
                labels = labels[:cut_off]
            units_dat.append(spks_data)
            units_labels.append(labels)

        data_all_parts.append(units_dat)
        labels_all_parts.append(units_labels)

    return data_all_parts, labels_all_parts


def exclude_neurons(data, session_list, path_locations, path_sel_exclude, loc=None, non_sig=False, threshold=0.0,
                    plot=False, times=np.linspace(-0.5, 2.0, 250), ylim=50):
    n_parts = len(data)
    data_new_ses_parts = []
    n_exc_parts = []
    n_neurons_parts = []
    exc_parts = []

    parts = session_list

    for p in range(n_parts):
        data_new_ses = []
        n_exc_ses = []
        n_neurons_ses = []
        for s in range(len(data[p])):
            data_ses = data[p][s]
            data_avg = data_ses.mean(0)
            mean_firing = data_avg.mean(1)
            if loc:
                cell_vals = pd.read_csv(path_locations.format(parts[p][s]))['Area'].values
                exc_idc = (cell_vals > loc[0]) & (cell_vals < loc[1])
                if threshold:
                    exc_thr = mean_firing > threshold
                    exc_idc = np.logical_and(exc_idc, exc_thr)
                    if non_sig:
                        sel_exc_idc = ~np.load(path_sel_exclude.format(parts[p][s]))
                        exc_idc = np.logical_and(exc_idc, sel_exc_idc)

            else:
                exc_idc = mean_firing > -1
            data_new = data_ses[:, exc_idc, :]
            data_new_ses.append(data_new)
            n_exc_ses.append(np.sum(exc_idc))
            n_neurons_ses.append(len(exc_idc))
        exc_proc = round(1 - np.sum(n_exc_ses) / np.sum(n_neurons_ses), 2)
        exc_parts.append(exc_proc)
        print('Excluded', exc_parts[p] * 100, '% of neurons from part', p + 1)
        if plot:
            colors = sns.xkcd_palette(['pale red'])
            data_avg_trl = [data_new_ses[_].mean(0) for _ in range(len(data_new_ses))]
            data_plot = np.concatenate(data_avg_trl, axis=0)
            df_plot = pd.DataFrame(data_plot.T, columns=list(range(1, data_plot.shape[0] + 1)))
            df_plot['Times'] = times
            df_melted = pd.melt(df_plot, id_vars=['Times'])
            df_melted['Rate'] = df_melted['value']
            df_melted['Neurons'] = df_melted['variable']
            plt.figure()
            sns.lineplot(data=df_melted, x="Times", y='Rate', hue="Neurons")
            plt.ylim([None, ylim])
            plt.axvline(0, linestyle="--", linedecode_epoch=0.8, color='black')
            sns.despine(right=True, top=True)
            plt.axvline(0.5, linestyle="--", linewidth=0.8, color='black')
            plt.axvline(1, linestyle="--", linewidth=0.8, color='black')
            mean_rate = df_melted.groupby('Neurons').mean()['Rate'].values
            plt.figure()
            sns.distplot(mean_rate, bins=20, kde=False, color='grey')
            sns.despine(right=True, top=True)
            median_firing_rate = round(np.median(data_plot.mean(1)), 2)
            plt.title("Median firing rate = " + str(median_firing_rate))
            plt.xlabel('Rate')
            plt.ylabel('Count')
            plt.axvline(median_firing_rate, 0, 1, linewidth=1, linestyle='--', color='black')
            exc_proc = round(1 - np.sum(n_exc_ses) / np.sum(n_neurons_ses), 2)
            plt.annotate('Exc % = ' + str(exc_proc), xy=(1, 40), color=colors[0])
            plt.xlim([None, 25])

        data_new_ses_parts.append(data_new_ses)

        n_neurons_parts.append(np.sum(n_neurons_ses))
        n_exc_parts.append(np.sum(n_neurons_ses) - np.sum(n_exc_ses))

    return data_new_ses_parts, exc_parts, n_neurons_parts


def condi_avg(data, labels):
    conditions = np.unique(labels)

    condition_averages = []
    for _ in range(len(conditions)):
        condition_dat = data[labels == conditions[_], :, :]
        condition_averages.append(condition_dat.mean(0))

    return np.array(condition_averages)


def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n, d = x.shape
    m, dy = y.shape
    assert (d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.))


def plot_covs(data, model_names, if_diff=False):
    n_models = len(data)
    n_epochs = len(data[0])
    if if_diff:
        covs = np.zeros((n_models, n_epochs + 1, 3, 3))
    else:
        covs = np.zeros((n_models, n_epochs, 3, 3))

    for m in range(n_models):
        for p in range(n_epochs):
            covs[m, p, :, :] = np.cov(data[m][p][:, :, 0].T)

    if if_diff:
        covs[:, n_epochs, :, :] = covs[:, 0, :, :] - covs[:, -1, :, :]
        n_epochs = n_epochs + 1

    max_cov = np.max(covs)
    min_cov = -max_cov
    print(min_cov)
    print(max_cov)
    norm = colors.TwoSlopeNorm(vmin=min_cov, vcenter=0, vmax=max_cov)
    covs_flat = np.resize(covs, (covs.shape[0] * covs.shape[1], covs.shape[2], covs.shape[3]))
    epoch_label = list(range(1, n_epochs + 1)) * 2

    fig, big_axes = plt.subplots(figsize=(8.0, 4.0), nrows=2, ncols=1, sharey=True)

    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(model_names[row - 1] + " \n", fontsize=14)

        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax.axis('off')
        # removes the white frame
        big_ax._frameon = False

    for i in range(1, n_epochs * 2 + 1):
        ax = fig.add_subplot(2, n_epochs, i)
        ax.imshow(covs_flat[i - 1, :, :], cmap=sns.diverging_palette(230, 20, as_cmap=True), norm=norm)
        if if_diff:
            if i == n_epochs or i == (n_epochs * 2):
                plt.title('difference (last vs first)')
            else:
                plt.title('epoch ' + str(epoch_label[i - 1]))
        else:
            plt.title('epoch ' + str(epoch_label[i - 1]))
        plt.axis('off')

    fig.set_facecolor('w')
    plt.tight_layout()
    plt.show()

    return covs


def get_betas_cross_val(data, labels, condition_labels, normalisation='mean_centred', time_window=[130, 150],
                        add_constant=True,
                        design_model='0/1', cross_val=False, if_xgen=False, task='task_1', full_model=False):
    n_parts = len(data)

    betas_part = []
    for p in range(n_parts):

        betas = []
        for s in range(len(data[p])):

            firing_rates_ses = data[p][s]
            firing_rates_ses = firing_rates_ses[:, :, time_window[0]:time_window[1]]
            labels_ses = labels[p][s]

            if task == 'task_1':
                firing_rates_ses = firing_rates_ses[labels_ses < 9, :, :]
                labels_ses = labels_ses[labels_ses < 9]
            elif task == 'task_2':
                firing_rates_ses = firing_rates_ses[labels_ses > 8, :, :]
                labels_ses = labels_ses[labels_ses > 8]

            if cross_val:
                firing_rates_ses1 = firing_rates_ses[::2, :, :]
                firing_rates_ses2 = firing_rates_ses[1::2, :, :]
                labels_ses1 = labels_ses[::2]
                labels_ses2 = labels_ses[1::2]
            else:
                firing_rates_ses1 = firing_rates_ses
                firing_rates_ses2 = firing_rates_ses
                labels_ses1 = labels_ses
                labels_ses2 = labels_ses

            if if_xgen:
                firing_rates_ses1 = firing_rates_ses[labels_ses1 < 9, :, :]
                firing_rates_ses2 = firing_rates_ses[labels_ses2 > 8, :, :]
                labels_ses1 = labels_ses1[labels_ses1 < 9]
                labels_ses2 = labels_ses2[labels_ses2 > 8]

            firing_rates_ses = [firing_rates_ses1, firing_rates_ses2]
            labels_ses = [labels_ses1, labels_ses2]

            betas_split = []
            n_splits = len(labels_ses)
            for split in range(n_splits):

                firing_mean = condi_avg(firing_rates_ses[split], labels_ses[split])

                if normalisation == 'zscore':
                    firing_mean = (firing_mean - np.mean(firing_mean, axis=0, keepdims=True)) / np.std(firing_mean,
                                                                                                       axis=0,
                                                                                                       keepdims=True)

                elif normalisation == 'mean_centred':
                    firing_mean = firing_mean - np.mean(firing_mean, axis=0, keepdims=True)

                elif normalisation == 'none':
                    firing_mean = firing_mean

                elif normalisation == 'soft':
                    firing_mean = firing_mean / (
                            (np.max(firing_mean, axis=0, keepdims=True) - np.min(firing_mean, axis=0,
                                                                                 keepdims=True)) + 5)

                if design_model == '0/1':
                    labels_cue = np.array(condition_labels[0])
                    labels_target = np.array(condition_labels[1])
                    labels_int1 = np.array(condition_labels[2])
                    if full_model:
                        labels_target2 = np.array(condition_labels[2])
                        labels_int1 = np.array(condition_labels[3])
                        labels_int2 = np.array(condition_labels[4])
                elif design_model == '+1/-1':
                    labels_cue = np.array(condition_labels[0])
                    labels_cue = np.where(labels_cue == 0, -1, labels_cue)
                    labels_target = np.array(condition_labels[1])
                    labels_target = np.where(labels_target == 0, -1, labels_target)
                    labels_int1 = np.array(condition_labels[2])
                    labels_int1 = np.where(labels_int1 == 0, -1, labels_int1)
                    if full_model:
                        labels_target2 = np.array(condition_labels[2])
                        labels_target2 = np.where(labels_target2 == 0, -1, labels_target2)
                        labels_int1 = np.array(condition_labels[3])
                        labels_int1 = np.where(labels_int1 == 0, -1, labels_int1)
                        labels_int2 = np.array(condition_labels[4])
                        labels_int2 = np.where(labels_int2 == 0, -1, labels_int2)

                if full_model:
                    if add_constant:
                        constant = np.ones_like(labels_cue).astype(float)
                        design_matrix = np.vstack(
                            [constant, labels_cue, labels_target, labels_int1, labels_target2, labels_int2]).T
                    else:
                        design_matrix = np.vstack(
                            [labels_cue, labels_target, labels_int1, labels_target2, labels_int2]).T
                else:
                    if add_constant:
                        constant = np.ones_like(labels_cue).astype(float)
                        design_matrix = np.vstack(
                            [constant, labels_cue, labels_target, labels_int1]).T
                    else:
                        design_matrix = np.vstack(
                            [labels_cue, labels_target, labels_int1]).T

                design_matrix = design_matrix.astype(float)

                n_neurons = firing_mean.shape[1]
                n_times = firing_mean.shape[-1]
                n_models = design_matrix.shape[1]

                design_matrix_stacked = np.concatenate([design_matrix] * n_times, axis=0)
                firing_rates_ses_stacked = np.concatenate(np.array_split(firing_mean, n_times, axis=-1), axis=0)

                betas_ses = np.zeros((n_neurons, n_models, 1))
                for cell in range(n_neurons):
                    betas_ses[cell, :, :] = \
                        sp.linalg.lstsq(design_matrix_stacked, firing_rates_ses_stacked[:, cell, :])[0][
                        :, :]

                betas_split.append(betas_ses)
            betas.append(np.array(betas_split)[:, :, :, 0])
        betas_part.append(betas)

    if full_model:

        epochs_rel = []
        for p in range(n_parts):
            if add_constant:
                epoch = np.concatenate(betas_part[p], axis=1)[:, :, 1:4].transpose(
                    (0, 2, 1))  # splits x coeffs x neurons
            else:
                epoch = np.concatenate(betas_part[p], axis=0)[:, :, :3].transpose((0, 2, 1))[0, :,
                        :]  # coeffs x neurons
            epoch -= np.mean(epoch, axis=2, keepdims=True)
            epochs_rel.append(epoch.T)

        epochs_irrel = []
        for p in range(n_parts):
            if add_constant:
                epoch_1 = np.concatenate(betas_part[p], axis=1)[:, :, 1][:, :, None].transpose(
                    (0, 2, 1))  # splits x coeffs x neurons
                epoch_2 = np.concatenate(betas_part[p], axis=1)[:, :, 4:].transpose(
                    (0, 2, 1))  # splits x coeffs x neurons
                epoch = np.concatenate([epoch_1, epoch_2], axis=1)
            else:
                epoch_1 = np.concatenate(betas_part[p], axis=1)[:, :, 0][:, :, None].transpose(
                    (0, 2, 1))  # splits x coeffs x neurons
                epoch_2 = np.concatenate(betas_part[p], axis=1)[:, :, 3:].transpose(
                    (0, 2, 1))  # splits x coeffs x neurons
                epoch = np.concatenate([epoch_1, epoch_2], axis=1)

            epoch -= np.mean(epoch, axis=2, keepdims=True)
            epochs_irrel.append(epoch.T)

        return epochs_rel, epochs_irrel

    else:
        epochs = []
        for p in range(n_parts):
            if add_constant:
                epoch = np.concatenate(betas_part[p], axis=1)[:, :, 1:4].transpose(
                    (0, 2, 1))  # splits x coeffs x neurons
            else:
                epoch = np.concatenate(betas_part[p], axis=0)[:, :, :3].transpose((0, 2, 1))[0, :,
                        :]  # coeffs x neurons
            epoch -= np.mean(epoch, axis=2, keepdims=True)
            epochs.append(epoch.T)

        return epochs


def get_freqs(x):
    return {value: len(list(freq)) for value, freq in groupby(sorted(list(x)))}


def dist_random(epochs, n_bootstraps=1000, rnd_model='gaussian (spherical)', model_names=['cue + shape', 'cue + width'],
                design_model='0/1', bon_correction=False, metric='KL divergance estimate', relative_dist=False):
    n_epochs = len(epochs[0])
    dfs = []
    p_vals = []
    for model_n in range(len(model_names)):

        KL = np.zeros((n_epochs, n_bootstraps))
        KL_r = np.zeros((n_epochs, n_bootstraps))
        KL_opt = np.zeros((n_epochs, n_bootstraps))
        for e in range(n_epochs):
            for bootstrap in tqdm(range(n_bootstraps)):
                data_train = epochs[model_n][e][:, :, 0]
                data_test = epochs[model_n][e][:, :, 1]

                if design_model == '0/1':
                    opt_cov = np.zeros((3, 3))
                    m = np.mean(np.diag(np.cov(data_train.T)))
                    opt_cov[:2, :2] = 0.5 * m
                    opt_cov[2, 2] = 2 * m
                    opt_cov[2, :2] = -m;
                    opt_cov[:2, 2] = -m

                elif design_model == '+1/-1':
                    # m = np.mean(np.diag(np.cov(data_train.T)))
                    # opt_cov = np.diag([0, 0, m * 3])
                    opt_cov = np.diag([0, 0, np.cov(data_train.T)[2, 2]])

                s_opt = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                      opt_cov,
                                                      data_train.shape[0])

                if rnd_model == 'gaussian (spherical)':
                    m = np.mean(np.diag(np.cov(data_train.T)))
                    shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                               np.diag([m, m, m]),
                                                               data_train.shape[0])
                    m = np.mean(np.diag(np.cov(data_test.T)))
                    shuffled_2 = np.random.multivariate_normal(np.zeros(data_test.shape[1]),
                                                               np.diag([m, m, m]),
                                                               data_test.shape[0])
                    if metric == 'KL divergance estimate':
                        kl_itr = 0.5 * (KLdivergence(data_test, shuffled_1) + KLdivergence(shuffled_1, data_test))
                        kl_r_itr = 0.5 * (KLdivergence(shuffled_2, shuffled_1) + KLdivergence(shuffled_1, shuffled_2))
                    elif metric == 'euclidean distance':
                        kl_itr = euclidean_distance(data_test, shuffled_1)
                        kl_r_itr = euclidean_distance(shuffled_2, shuffled_1)
                    elif metric == 'epairs':
                        kl_itr = epairs_metric(data_test, shuffled_1)
                        kl_r_itr = epairs_metric(shuffled_2, shuffled_1)

                elif rnd_model == 'gaussian (tied)':

                    shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                               np.diag(np.diag(np.cov(data_train.T))),
                                                               data_train.shape[0])

                    shuffled_2 = np.random.multivariate_normal(np.zeros(data_test.shape[1]),
                                                               np.diag(np.diag(np.cov(data_test.T))),
                                                               data_test.shape[0])

                    if metric == 'KL divergance estimate':
                        kl_itr = 0.5 * (KLdivergence(data_test, shuffled_1) + KLdivergence(shuffled_1, data_test))
                        kl_r_itr = 0.5 * (KLdivergence(shuffled_2, shuffled_1) + KLdivergence(shuffled_1, shuffled_2))
                    elif metric == 'euclidean distance':
                        kl_itr = euclidean_distance(data_test, shuffled_1)
                        kl_r_itr = euclidean_distance(shuffled_2, shuffled_1)
                    elif metric == 'epairs':
                        kl_itr = epairs_metric(data_test, shuffled_1)
                        kl_r_itr = epairs_metric(shuffled_2, shuffled_1)

                KL[e, bootstrap] = kl_itr
                KL_r[e, bootstrap] = kl_r_itr

                if metric == 'KL divergance estimate':
                    KL_opt[e, bootstrap] = 0.5 * (KLdivergence(s_opt, shuffled_1) + KLdivergence(shuffled_1, s_opt))
                elif metric == 'euclidean distance':
                    KL_opt[e, bootstrap] = euclidean_distance(s_opt, shuffled_1)
                elif metric == 'epairs':
                    KL_opt[e, bootstrap] = epairs_metric(s_opt, shuffled_1)

        if relative_dist:
            KL_r_avg = np.mean(KL_r, keepdims=True, axis=-1)
            KL -= KL_r_avg
            KL_r -= KL_r_avg
            KL_opt -= KL_r_avg
            KL_opt_avg = np.mean(KL_opt, keepdims=True, axis=-1)
            KL /= KL_opt_avg
            KL_r /= KL_opt_avg
            KL_opt /= KL_opt_avg

        p = 1 * (np.sum(KL_r >= np.mean(KL, axis=-1, keepdims=True), axis=-1) / n_bootstraps)
        if bon_correction:
            p = p * n_epochs
        p_vals.append(p)

        epoch = np.sort(list(range(1, n_epochs + 1)) * n_bootstraps)
        epoch_labels = np.concatenate([epoch, epoch, epoch])
        dist_label = np.concatenate([['observed'] * (n_bootstraps * n_epochs), [rnd_model] * (n_bootstraps * n_epochs),
                                     ['structured'] * (n_bootstraps * n_epochs)])
        KL_df = np.reshape(KL, KL.shape[0] * KL.shape[1])
        KL_r_df = np.reshape(KL_r, KL_r.shape[0] * KL_r.shape[1])
        KL_opt_df = np.reshape(KL_opt, KL_opt.shape[0] * KL_opt.shape[1])
        KL_all_df = np.concatenate([KL_df, KL_r_df, KL_opt_df])

        df = pd.DataFrame(np.array([KL_all_df, epoch_labels, dist_label]).T,
                          columns=[metric, 'learning epoch', 'distribution'])
        df[metric] = df[metric].astype(float)
        df['model'] = model_names[model_n]

        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all['divergence from'] = 'random selectivity'
    print(p_vals)
    return df_all, p_vals, [KL, KL_r, KL_opt]


def dist_structured(epochs, n_bootstraps=1000, rnd_model='gaussian (spherical)',
                    model_names=['cue + shape', 'cue + width'],
                    design_model='0/1', bon_correction=False, metric='KL divergance estimate', relative_dist=False):
    n_epochs = len(epochs[0])
    dfs = []
    p_vals = []

    for model_n in range(len(model_names)):

        KL = np.zeros((n_epochs, n_bootstraps))
        KL_r = np.zeros((n_epochs, n_bootstraps))
        KL_opt = np.zeros((n_epochs, n_bootstraps))
        for e in range(n_epochs):
            for bootstrap in tqdm(range(n_bootstraps)):

                data_train = epochs[model_n][e][:, :, 0]
                data_test = epochs[model_n][e][:, :, 1]

                if design_model == '0/1':

                    opt_cov1 = np.zeros((3, 3))
                    m = np.mean(np.diag(np.cov(data_train.T)))
                    opt_cov1[:2, :2] = 0.5 * m
                    opt_cov1[2, 2] = 2 * m
                    opt_cov1[2, :2] = -m;
                    opt_cov1[:2, 2] = -m

                    opt_cov2 = np.zeros((3, 3))
                    m = np.mean(np.diag(np.cov(data_test.T)))
                    opt_cov2[:2, :2] = 0.5 * m
                    opt_cov2[2, 2] = 2 * m
                    opt_cov2[2, :2] = -m;
                    opt_cov2[:2, 2] = -m

                elif design_model == '+1/-1':

                    opt_cov1 = np.diag([0, 0, np.cov(data_train.T)[2, 2]])
                    opt_cov2 = np.diag([0, 0, np.cov(data_test.T)[2, 2]])

                s_opt_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                        opt_cov1,
                                                        data_train.shape[0])

                s_opt_2 = np.random.multivariate_normal(np.zeros(data_test.shape[1]),
                                                        opt_cov2,
                                                        data_test.shape[0])

                if rnd_model == 'gaussian (tied)':
                    shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                               np.diag(np.diag(np.cov(data_train.T))),
                                                               data_train.shape[0])

                    if metric == 'KL divergance estimate':
                        KL[e, bootstrap] = 0.5 * (KLdivergence(data_test, s_opt_1) + KLdivergence(s_opt_1, data_test))
                        KL_r[e, bootstrap] = 0.5 * (
                                KLdivergence(shuffled_1, s_opt_1) + KLdivergence(s_opt_1, shuffled_1))
                        KL_opt[e, bootstrap] = 0.5 * (KLdivergence(s_opt_2, s_opt_1) + KLdivergence(s_opt_1, s_opt_2))
                    elif metric == 'euclidean distance':
                        KL[e, bootstrap] = euclidean_distance(data_test, s_opt_1)
                        KL_r[e, bootstrap] = euclidean_distance(shuffled_1, s_opt_1)
                        KL_opt[e, bootstrap] = euclidean_distance(s_opt_2, s_opt_1)
                    elif metric == 'epairs':
                        KL[e, bootstrap] = epairs_metric(data_test, s_opt_1)
                        KL_r[e, bootstrap] = epairs_metric(shuffled_1, s_opt_1)
                        KL_opt[e, bootstrap] = epairs_metric(s_opt_2, s_opt_1)

                if rnd_model == 'gaussian (spherical)':
                    m = np.mean(np.diag(np.cov(data_train.T)))
                    shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                               np.diag([m, m, m]),
                                                               data_train.shape[0])
                    if metric == 'KL divergance estimate':
                        KL[e, bootstrap] = 0.5 * (KLdivergence(data_test, s_opt_1) + KLdivergence(s_opt_1, data_test))
                        KL_r[e, bootstrap] = 0.5 * (
                                KLdivergence(shuffled_1, s_opt_1) + KLdivergence(s_opt_1, shuffled_1))
                        KL_opt[e, bootstrap] = 0.5 * (KLdivergence(s_opt_2, s_opt_1) + KLdivergence(s_opt_1, s_opt_2))

                    elif metric == 'euclidean distance':
                        KL[e, bootstrap] = euclidean_distance(data_test, s_opt_1)
                        KL_r[e, bootstrap] = euclidean_distance(shuffled_1, s_opt_1)
                        KL_opt[e, bootstrap] = euclidean_distance(s_opt_2, s_opt_1)

                    elif metric == 'epairs':
                        KL[e, bootstrap] = epairs_metric(data_test, s_opt_1)
                        KL_r[e, bootstrap] = epairs_metric(shuffled_1, s_opt_1)
                        KL_opt[e, bootstrap] = epairs_metric(s_opt_2, s_opt_1)

        if relative_dist:
            KL_opt_avg = np.mean(KL_opt, keepdims=True, axis=-1)
            KL -= KL_opt_avg
            KL_r -= KL_opt_avg
            KL_opt -= KL_opt_avg
            KL_r_avg = np.mean(KL_r, keepdims=True, axis=-1)
            KL /= KL_r_avg
            KL_r /= KL_r_avg
            KL_opt /= KL_r_avg

        p = 1 * (np.sum(KL_r <= np.mean(KL, axis=-1, keepdims=True), axis=-1) / n_bootstraps)
        if bon_correction:
            p = p * n_epochs

        p_vals.append(p)
        epoch = np.sort(list(range(1, n_epochs + 1)) * n_bootstraps)
        epoch_labels = np.concatenate([epoch, epoch, epoch])
        dist_label = np.concatenate([['observed'] * (n_bootstraps * n_epochs),
                                     [rnd_model] * (n_bootstraps * n_epochs),
                                     ['structured'] * (n_bootstraps * n_epochs)])
        KL_df = np.reshape(KL, KL.shape[0] * KL.shape[1])
        KL_r_df = np.reshape(KL_r, KL_r.shape[0] * KL_r.shape[1])
        KL_opt_df = np.reshape(KL_opt, KL_opt.shape[0] * KL_opt.shape[1])
        KL_all_df = np.concatenate([KL_df, KL_r_df, KL_opt_df])
        df = pd.DataFrame(np.array([KL_all_df, epoch_labels, dist_label]).T,
                          columns=[metric, 'learning epoch', 'distribution'])
        df[metric] = df[metric].astype(float)
        df['model'] = model_names[model_n]

        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all['divergence from'] = 'structured selectivity'
    print(p_vals)

    return df_all, p_vals, [KL, KL_r, KL_opt]


def fit_lines(data1, data2, if_xgen=False):
    r_best_n, r_opt_n = [], []
    for side in range(2):

        if side == 0:
            data_train = data1
            data_test = data2
        elif side == 1:
            data_train = data2
            data_test = data1

        N = data_train.shape[0]
        # colour as a regressor
        A1 = np.ones((N, 2))  # With intercept/constant
        A1[:, 1] = data_train[:, 0]  # Use color selectivities as the regressor
        x_shape_c, _, _, _ = np.linalg.lstsq(A1, data_train[:, 1], rcond=-1)  # Fit to the shape selectivies
        x_interaction_c, _, _, _ = np.linalg.lstsq(A1, data_train[:, 2], rcond=-1)  # Fit to the interaction selectivies
        A1[:, 1] = data_test[:, 0]  # Use color selectivities as the regressor
        pred_best1 = np.array([A1[:, 1] * x_shape_c[1], A1[:, 1] * x_interaction_c[1]]).T
        r_best1 = r2_score(data_test[:, 1:], pred_best1)

        # shape as a regressor
        A2 = np.ones((N, 2))  # With intercept/constant
        A2[:, 1] = data_train[:, 1]  # Use shape selectivities as the regressor
        x_colour_s, _, _, _ = np.linalg.lstsq(A2, data_train[:, 0], rcond=-1)  # Fit to the colour selectivies
        x_interaction_s, _, _, _ = np.linalg.lstsq(A2, data_train[:, 2], rcond=-1)  # Fit to the interaction selectivies
        A2[:, 1] = data_test[:, 1]  # Use color selectivities as the regressor
        pred_best2 = np.array([A2[:, 1] * x_colour_s[1], A2[:, 1] * x_interaction_s[1]]).T
        data_compare = np.concatenate([data_test[:, 0][:, None], data_test[:, -1][:, None]], axis=1)
        r_best2 = r2_score(data_compare, pred_best2)

        # interaction as a regressor
        A3 = np.ones((N, 2))  # With intercept/constant
        A3[:, 1] = data_train[:, 2]  # Use interaction selectivities as the regressor
        x_colour_int, _, _, _ = np.linalg.lstsq(A3, data_train[:, 0], rcond=-1)  # Fit to the colour selectivies
        x_shape_int, _, _, _ = np.linalg.lstsq(A3, data_train[:, 1], rcond=-1)  # Fit to the shape selectivies
        A3[:, 1] = data_test[:, 2]  # Use interaction selectivities as the regressor
        pred_best3 = np.array([A3[:, 1] * x_colour_int[1], A3[:, 1] * x_shape_int[1]]).T
        data_compare = np.concatenate([data_test[:, 0][:, None], data_test[:, 1][:, None]], axis=1)
        r_best3 = r2_score(data_compare, pred_best3)

        if if_xgen:
            pred_opt1 = np.zeros((data_train.shape[0], 2))
            pred_opt1[:, 0] = data_train[:, 0]
            pred_opt1[:, 1] = -2 * data_train[:, 0]
            r_opt1 = r2_score(data_test[:, 1:], pred_opt1)

        else:
            pred_opt1 = np.zeros((data_test.shape[0], 2))
            pred_opt1[:, 0] = data_test[:, 0]
            pred_opt1[:, 1] = -2 * data_test[:, 0]
            r_opt1 = r2_score(data_test[:, 1:], pred_opt1)

        if if_xgen:
            pred_opt2 = np.zeros((data_train.shape[0], 2))
            pred_opt2[:, 0] = data_train[:, 1]
            pred_opt2[:, 1] = -2 * data_train[:, 1]
            data_compare = np.concatenate([data_test[:, 0][:, None], data_test[:, -1][:, None]], axis=1)
            r_opt2 = r2_score(data_compare, pred_opt2)

        else:
            pred_opt2 = np.zeros((data_test.shape[0], 2))
            pred_opt2[:, 0] = data_test[:, 1]
            pred_opt2[:, 1] = -2 * data_test[:, 1]
            data_compare = np.concatenate([data_test[:, 0][:, None], data_test[:, -1][:, None]], axis=1)
            r_opt2 = r2_score(data_compare, pred_opt2)

        if if_xgen:
            pred_opt3 = np.zeros((data_train.shape[0], 2))
            pred_opt3[:, 0] = data_train[:, -1] / -2
            pred_opt3[:, 1] = data_train[:, -1] / -2
            data_compare = np.concatenate([data_test[:, 0][:, None], data_test[:, 1][:, None]], axis=1)
            r_opt3 = r2_score(data_compare, pred_opt3)

        else:
            pred_opt3 = np.zeros((data_test.shape[0], 2))
            pred_opt3[:, 0] = data_test[:, -1] / -2
            pred_opt3[:, 1] = data_test[:, -1] / -2
            data_compare = np.concatenate([data_test[:, 0][:, None], data_test[:, 1][:, None]], axis=1)
            r_opt3 = r2_score(data_compare, pred_opt3)

        r_best_n.append(np.mean([r_best1, r_best2, r_best3]))
        r_opt_n.append(np.mean([r_opt1, r_opt2, r_opt3]))

    r_best = np.mean(r_best_n)
    r_opt = np.mean(r_opt_n)

    return r_best, r_opt


def r_squared_pop(epochs, rnd_model='gaussian (spherical)', n_bootstraps=1000,
                  model_names=['cue + shape', 'cue + width'], break_axis=1.5, if_xgen=False):
    dfs = []
    n_epochs = len(epochs[0])
    for model_n in range(len(model_names)):

        fits_data = np.zeros((n_epochs, 2))
        fits_opt = np.zeros((n_bootstraps, n_epochs, 2))
        fits_rnd = np.zeros((n_bootstraps, n_epochs, 2))
        for e in range(n_epochs):

            s_train = epochs[model_n][e][:, :, 0]
            s_test = epochs[model_n][e][:, :, 1]

            # remove mean across neurons so we don't have to fit an intercept
            s_train -= np.mean(s_train, axis=0, keepdims=True)  # length 3 vector
            s_test -= np.mean(s_test, axis=0, keepdims=True)  # length 3 vector

            fits_data[e, :] = fit_lines(s_train, s_test, if_xgen=if_xgen)

            for bootstrap in tqdm(range(n_bootstraps)):

                opt_cov = np.eye(3)
                m = np.mean(np.diag(np.cov(s_train.T)))
                opt_cov[:2, :2] = 0.5 * m
                opt_cov[2, 2] = 2 * m
                opt_cov[2, :2] = -m
                opt_cov[:2, 2] = -m

                opt_cov = np.eye(3)
                m = np.mean(np.diag(np.cov(s_test.T)))
                opt_cov[:2, :2] = 0.5 * m
                opt_cov[2, 2] = 2 * m
                opt_cov[2, :2] = -m
                opt_cov[:2, 2] = -m

                s_train_opt = np.random.multivariate_normal(np.zeros(s_train.shape[1]),
                                                            opt_cov,
                                                            s_train.shape[0])

                s_test_opt = np.random.multivariate_normal(np.zeros(s_test.shape[1]),
                                                           opt_cov,
                                                           s_test.shape[0])

                fits_opt[bootstrap, e, :] = fit_lines(s_train_opt, s_test_opt)
                fits_opt[bootstrap, e, 0] += 0.05
                fits_opt[bootstrap, e, 1] -= 0.05

                if rnd_model == 'gaussian (spherical)':
                    m = np.mean(np.diag(np.cov(s_train.T)))
                    s_train_rnd = np.random.multivariate_normal(np.zeros(s_train.shape[1]),
                                                                np.diag([m, m, m]),
                                                                s_train.shape[0])
                    m = np.mean(np.diag(np.cov(s_test.T)))
                    s_test_rnd = np.random.multivariate_normal(np.zeros(s_test.shape[1]),
                                                               np.diag([m, m, m]),
                                                               s_test.shape[0])
                elif rnd_model == 'gaussian (tied)':

                    s_train_rnd = np.random.multivariate_normal(np.zeros(s_train.shape[1]),
                                                                np.diag(np.diag(np.cov(s_train.T))),
                                                                s_train.shape[0])

                    s_test_rnd = np.random.multivariate_normal(np.zeros(s_test.shape[1]),
                                                               np.diag(np.diag(np.cov(s_test.T))),
                                                               s_test.shape[0])

                fits_rnd[bootstrap, e, :] = fit_lines(s_train_rnd, s_test_rnd)
                if break_axis:
                    fits_rnd[bootstrap, e, 1] += break_axis

        epoch_data = list(range(1, n_epochs + 1)) * 2
        epoch = np.concatenate([np.sort(list(range(1, n_epochs + 1)) * n_bootstraps)] * 2)
        epoch_labels = np.concatenate([epoch_data, epoch, epoch])
        gen_model = np.concatenate([['data'] * (n_epochs * 2),
                                    ['gaussian'] * (n_epochs * 2 * n_bootstraps),
                                    ['optimal'] * (n_epochs * 2 * n_bootstraps),
                                    ])

        dist_label = np.concatenate([['best fit line'] * n_epochs,
                                     ['optimal XOR line'] * n_epochs,
                                     ['best fit line'] * (n_epochs * n_bootstraps),
                                     ['optimal XOR line'] * (n_epochs * n_bootstraps),
                                     ['best fit line'] * (n_epochs * n_bootstraps),
                                     ['optimal XOR line'] * (n_epochs * n_bootstraps),
                                     ])

        data_df = np.reshape(fits_data, fits_data.shape[0] * fits_data.shape[1], order='F')
        rnd_df = np.reshape(fits_rnd, fits_rnd.shape[0] * fits_rnd.shape[1] * fits_rnd.shape[2], order='F')
        opt_df = np.reshape(fits_opt, fits_opt.shape[0] * fits_opt.shape[1] * fits_opt.shape[2], order='F')

        all_df = np.concatenate([data_df, rnd_df, opt_df])
        df = pd.DataFrame(np.array([all_df, epoch_labels, dist_label, gen_model]).T,
                          columns=['r squared', 'learning epoch', 'fitted model', 'generative model'])
        df['r squared'] = df['r squared'].astype(float)
        df['model'] = model_names[model_n]

        dfs.append(df)

    return pd.concat(dfs)


def xor_sim_noise(sig_min, sig_max, sig_n, n_neurons=400):
    s_list = np.linspace(sig_min, sig_max, sig_n)
    s_tot = s_list.shape[0]
    N = n_neurons
    n_coeffs = 3

    conds = np.ones((4, n_coeffs))
    conds[0, :2] = -1
    conds[1, [0, 2]] = -1
    conds[2, 1:] = -1

    targets = [1, -1, -1, 1]  # XOR
    clf1 = linear_model.LogisticRegression()  # 1 is no reg

    n = 100
    perf = np.zeros((s_tot, n, 2))
    print('XOr decoding performance as a function of noice (sigma) - running simulation ')
    for s_c, sig in enumerate(s_list):
        print(s_c, end=',')
        for i in range(n):
            # Random selectivity
            cov = np.eye(3)
            s = np.random.multivariate_normal(np.zeros(3), cov, N)
            r = s @ conds.T
            r_train = r.T + sig * np.random.normal(0, 1, (4, N))
            r_train = r_train - np.mean(r_train, axis=1, keepdims=True)
            clf1.fit(r_train, targets)
            r_test = r.T + sig * np.random.normal(0, 1, (4, N))
            r_test = r_test - np.mean(r_test, axis=1, keepdims=True)
            perf[s_c, i, 0] = clf1.score(r_test, targets)

            # Structured selectivity
            opt_cov = np.diag([0, 0, 3])
            s = np.random.multivariate_normal(np.zeros(3), opt_cov, N)

            r = s @ conds.T
            r_train = r.T + sig * np.random.normal(0, 1, (4, N))
            r_train = r_train - np.mean(r_train, axis=1, keepdims=True)
            clf1.fit(r_train, targets)
            r_test = r.T + sig * np.random.normal(0, 1, (4, N))
            r_test = r_test - np.mean(r_test, axis=1, keepdims=True)
            perf[s_c, i, 1] = clf1.score(r_test, targets)

    return perf, s_list


def xor_sim_units(n_neurons, step_neurons, noise_sig):
    N_list = np.arange(1, n_neurons, step_neurons)
    N_tot = N_list.shape[0]

    conds = np.ones((4, 3))
    conds[0, :2] = -1
    conds[1, [0, 2]] = -1
    conds[2, 1:] = -1

    targets = [1, -1, -1, 1]  # XOR
    clf1 = linear_model.LogisticRegression()  # 1 is no reg
    sig = noise_sig

    n = 100
    perf_units = np.zeros((N_tot, n, 2))
    for N_c, N in enumerate(N_list):
        print(N, end=',')
        for i in range(n):
            s = np.random.normal(0, 1, (N, 3))
            r = s @ conds.T
            clf1.fit(r.T + sig * np.random.normal(0, 1, (4, N)), targets)
            perf_units[N_c, i, 0] = clf1.score(r.T + sig * np.random.normal(0, 1, (4, N)), targets)

            # Structured selectivity
            opt_cov = np.diag([0, 0, 3])
            s = np.random.multivariate_normal(np.zeros(3), opt_cov, N)

            r = s @ conds.T
            clf1.fit(r.T + sig * np.random.normal(0, 1, (4, N)), targets)
            perf_units[N_c, i, 1] = clf1.score(r.T + sig * np.random.normal(0, 1, (4, N)), targets)

    return perf_units, N_list


def generate_boundary(sel, sig=0.9, n=100):
    conds = np.ones((4, 3))
    conds[0, :2] = -1
    conds[1, [0, 2]] = -1
    conds[2, 1:] = -1
    labels = [1, -1, -1, 1]  # XOR

    neurons = sel[:2, :]  # take the first two neurons
    rates = neurons @ conds.T
    rates = np.concatenate([rates] * n, axis=-1)
    rates = rates + np.random.normal(0, sig, (2, 4 * n))
    rates -= np.mean(rates, axis=1, keepdims=True)
    labels = np.concatenate([labels] * n, axis=0)

    clf = LDA()
    clf.fit(rates.T, labels)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-10, 10)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    return rates[:, :4], xx, yy


def KL_optimal_2(epochs, n_bootstraps=1000, rnd_model='gaussian (tied)', model_names=['cue + shape', 'cue + width'],
                 bon_correction=False):
    n_epochs = len(epochs[0])
    dfs = []
    p_vals = []

    for model_n in range(len(model_names)):

        KL = np.zeros((n_epochs, n_bootstraps))
        KL_r = np.zeros((n_epochs, n_bootstraps))
        for e in range(n_epochs):
            for bootstrap in tqdm(range(n_bootstraps)):

                data_train = epochs[model_n][e][:, :, 0]
                data_test = epochs[model_n][e][:, :, 1]

                s_opt = np.zeros(data_test.shape)
                s_opt[:, 0] = data_test[:, 0]
                s_opt[:, 1] = data_test[:, 0]
                s_opt[:, 2] = data_test[:, 0] * -2

                if rnd_model == 'gaussian (tied)':
                    shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                               np.diag(np.diag(np.cov(data_train.T))),
                                                               data_train.shape[0])

                    KL[e, bootstrap] = 0.5 * (KLdivergence(data_test, s_opt) + KLdivergence(s_opt, data_test))
                    KL_r[e, bootstrap] = 0.5 * (KLdivergence(shuffled_1, s_opt) + KLdivergence(s_opt, shuffled_1))

                if rnd_model == 'gaussian (spherical)':
                    m = np.mean(np.diag(np.cov(data_train.T)))
                    shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                               np.diag([m, m, m]),
                                                               data_train.shape[0])

                    KL[e, bootstrap] = 0.5 * (KLdivergence(data_train, s_opt) + KLdivergence(s_opt, data_train))
                    KL_r[e, bootstrap] = 0.5 * (KLdivergence(shuffled_1, s_opt) + KLdivergence(s_opt, shuffled_1))

        p = 2 * (np.sum(KL_r <= np.mean(KL, axis=-1, keepdims=True), axis=-1) / n_bootstraps)
        if bon_correction:
            p = p * n_epochs

        p_vals.append(p)
        epoch = np.sort(list(range(1, n_epochs + 1)) * n_bootstraps)
        epoch_labels = np.concatenate([epoch, epoch])
        dist_label = np.concatenate([['observed'] * (n_bootstraps * n_epochs), [rnd_model] * (n_bootstraps * n_epochs)])
        KL_df = np.reshape(KL, KL.shape[0] * KL.shape[1])
        KL_r_df = np.reshape(KL_r, KL_r.shape[0] * KL_r.shape[1])
        KL_all_df = np.concatenate([KL_df, KL_r_df])
        df = pd.DataFrame(np.array([KL_all_df, epoch_labels, dist_label]).T,
                          columns=['KL divergance estimate', 'learning epoch', 'distribution'])
        df['KL divergance estimate'] = df['KL divergance estimate'].astype(float)
        df['model'] = model_names[model_n]

        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all['divergence from'] = 'optimal selectivity'
    print(p_vals)

    return df_all, p_vals


def plot_pvals(df_data, p, ax, offset=0.0001, tail=1, n_bootstraps=1000, metric='KL divergance estimate'):
    y_vals = np.resize(df_data[df_data["distribution"] == 'observed'][metric].values,
                       (len(p), n_bootstraps))
    y_avgs = y_vals.mean(-1)
    y_stds = y_vals.std(-1)
    if tail == 1:
        y_pos = y_avgs + y_stds + offset
    elif tail == -1:
        y_pos = y_avgs - y_stds - (offset * 12)

    for e in range(len(p)):
        if p[e] > 0.05:
            star = 'ns'
            size = 10
        elif (p[e] <= 0.05) & (p[e] > 0.01):
            star = '*'
            size = 14
        elif (p[e] <= 0.01) & (p[e] > 0.001):
            star = '**'
            size = 14
        elif p[e] <= 0.001:
            star = '***'
            size = 14
        ax.text(e, y_pos[e], star, ha='center', size=size)

    return


def get_betas_cross_val_2(data, labels, condition_labels, normalisation='zscore', time_window=[140, 150],
                          cross_val=False, if_xgen=False, task='task_1', shuffle=False):
    n_parts = len(data)

    betas_part = []
    for p in range(n_parts):

        betas = []
        for s in range(len(data[p])):

            firing_rates_ses = data[p][s]
            firing_rates_ses = firing_rates_ses[:, :, time_window[0]:time_window[1]]
            labels_ses = labels[p][s]

            if task == 'task_1':
                firing_rates_ses = firing_rates_ses[labels_ses < 9, :, :]
                labels_ses = labels_ses[labels_ses < 9]
            elif task == 'task_2':
                firing_rates_ses = firing_rates_ses[labels_ses > 8, :, :]
                labels_ses = labels_ses[labels_ses > 8]
            elif task == 'context_1':
                fac_cxt1 = np.array([1, 2, 3, 4, 9, 10, 11, 12])
                idc_cxt1 = np.isin(labels_ses, fac_cxt1)
                firing_rates_ses = firing_rates_ses[idc_cxt1, :, :]
                labels_ses = labels_ses[idc_cxt1]
            elif task == 'context_2':
                fac_cxt2 = np.array([5, 6, 7, 8, 13, 14, 15, 16])
                idc_cxt2 = np.isin(labels_ses, fac_cxt2)
                firing_rates_ses = firing_rates_ses[idc_cxt2, :, :]
                labels_ses = labels_ses[idc_cxt2]

            if cross_val:
                n_trls = firing_rates_ses.shape[0]
                idc_0 = np.zeros(n_trls // 2)
                idc_1 = np.ones(n_trls // 2)

                if (n_trls % 2) > 0:
                    idc_1 = np.concatenate([idc_1, [1.0]])

                idc_rnd = np.concatenate([idc_0, idc_1])

                random.shuffle(idc_rnd)
                firing_rates_ses1 = firing_rates_ses[idc_rnd < 1, :, :]
                firing_rates_ses2 = firing_rates_ses[idc_rnd > 0, :, :]
                labels_ses1 = labels_ses[idc_rnd < 1]
                labels_ses2 = labels_ses[idc_rnd > 0]
            else:
                firing_rates_ses1 = firing_rates_ses
                firing_rates_ses2 = firing_rates_ses
                labels_ses1 = labels_ses
                labels_ses2 = labels_ses

            if if_xgen:
                firing_rates_ses1 = firing_rates_ses[labels_ses1 < 9, :, :]
                firing_rates_ses2 = firing_rates_ses[labels_ses2 > 8, :, :]
                labels_ses1 = labels_ses1[labels_ses1 < 9]
                labels_ses2 = labels_ses2[labels_ses2 > 8]

            firing_rates_ses = [firing_rates_ses1, firing_rates_ses2]
            labels_ses = [labels_ses1, labels_ses2]

            betas_split = []
            n_splits = len(labels_ses)
            for split in range(n_splits):

                firing_mean = np.mean(firing_rates_ses[split], axis=-1, keepdims=True)
                labels_ses_split = labels_ses[split]

                if shuffle:
                    random.shuffle(labels_ses_split)

                if normalisation == 'zscore':
                    firing_mean = (firing_mean - np.mean(firing_mean, axis=0, keepdims=True)) / (
                        np.std(firing_mean, axis=0,
                               keepdims=True))

                elif normalisation == 'mean_centred':
                    firing_mean = firing_mean - np.mean(firing_mean, axis=0, keepdims=True)

                elif normalisation == 'none':
                    firing_mean = firing_mean

                elif normalisation == 'soft':
                    firing_mean = firing_mean / (
                            (np.max(firing_mean, axis=0, keepdims=True) - np.min(firing_mean, axis=0,

                                                                                 keepdims=True)) + 5)

                labels_cue = np.array(assign_lables(labels_ses_split, condition_labels[0]))
                labels_target = np.array(assign_lables(labels_ses_split, condition_labels[1]))
                labels_int1 = np.array(assign_lables(labels_ses_split, condition_labels[2]))
                labels_target2 = np.array(assign_lables(labels_ses_split, condition_labels[3]))
                labels_int2 = np.array(assign_lables(labels_ses_split, condition_labels[4]))
                constant = np.ones_like(labels_cue).astype(float)
                design_matrix = np.vstack(
                    [constant, labels_cue, labels_target, labels_int1, labels_target2, labels_int2]).T

                if task == 'all':
                    labels_context = np.array(assign_lables(labels_ses_split, condition_labels[0]))
                    labels_shape = np.array(assign_lables(labels_ses_split, condition_labels[1]))
                    labels_rew = np.array(assign_lables(labels_ses_split, condition_labels[2]))
                    labels_task = np.array(assign_lables(labels_ses_split, condition_labels[3]))
                    labels_width = np.array(assign_lables(labels_ses_split, condition_labels[4]))
                    labels_int_irr = np.array(assign_lables(labels_ses_split, condition_labels[5]))
                    constant = np.ones_like(labels_cue).astype(float)
                    design_matrix = np.vstack(
                        [constant, labels_context, labels_shape, labels_rew, labels_task, labels_width,
                         labels_int_irr]).T

                design_matrix = design_matrix.astype(float)

                n_neurons = firing_mean.shape[1]
                n_times = firing_mean.shape[-1]
                n_models = design_matrix.shape[1]

                design_matrix_stacked = np.concatenate([design_matrix] * n_times, axis=0)
                firing_rates_ses_stacked = np.concatenate(np.array_split(firing_mean, n_times, axis=-1), axis=0)

                betas_ses = np.zeros((n_neurons, n_models, 1))
                for cell in range(n_neurons):
                    betas_ses[cell, :, :] = \
                        sp.linalg.lstsq(design_matrix_stacked, firing_rates_ses_stacked[:, cell, :])[0][:, :]

                betas_split.append(betas_ses)
            betas.append(np.array(betas_split)[:, :, :, 0])
        betas_part.append(betas)

    if task == 'all':
        epochs_rel = []
        for p in range(n_parts):
            epoch = np.concatenate(betas_part[p], axis=1)[:, :, 1:4].transpose((0, 2, 1))  # splits x coeffs x neurons
            epoch -= np.mean(epoch, axis=2, keepdims=True)
            epochs_rel.append(epoch.T)

        epochs_irrel = []
        for p in range(n_parts):
            epoch = np.concatenate(betas_part[p], axis=1)[:, :, 4:].transpose((0, 2, 1))  # splits x coeffs x neurons
            epoch -= np.mean(epoch, axis=2, keepdims=True)
            epochs_irrel.append(epoch.T)

    else:
        epochs_rel = []
        for p in range(n_parts):
            epoch = np.concatenate(betas_part[p], axis=1)[:, :, 1:4].transpose((0, 2, 1))  # splits x coeffs x neurons
            epoch -= np.mean(epoch, axis=2, keepdims=True)
            epochs_rel.append(epoch.T)

        epochs_irrel = []
        for p in range(n_parts):
            epoch_1 = np.concatenate(betas_part[p], axis=1)[:, :, 1][:, :, None].transpose(
                (0, 2, 1))  # splits x coeffs x neurons
            epoch_2 = np.concatenate(betas_part[p], axis=1)[:, :, 4:].transpose((0, 2, 1))  # splits x coeffs x neurons
            epoch = np.concatenate([epoch_1, epoch_2], axis=1)
            epoch -= np.mean(epoch, axis=2, keepdims=True)
            epochs_irrel.append(epoch.T)

    return epochs_rel, epochs_irrel


def get_betas_cross_val_3(data, labels, condition_labels, normalisation='zscore', time_window=[140, 150],
                          add_constant=True, n_splits=10):
    n_parts = len(data)
    betas_part = []
    for p in range(n_parts):

        betas_ses_all = []
        for s in tqdm(range(len(data[p]))):

            firing_rates_ses = data[p][s]
            firing_rates_ses = firing_rates_ses[:, :, time_window[0]:time_window[1]]
            labels_ses = labels[p][s]

            n_trl = firing_rates_ses.shape[0]
            firing_rates_ses_all = []
            labels_ses_all = []
            for split in range(n_splits):

                idc_1 = np.zeros(n_trl // 2)
                idc_2 = np.ones(n_trl // 2)
                if (n_trl % 2) > 0:
                    idc_2 = np.append(idc_2, [1.0])
                idc_rnd = np.concatenate([idc_1, idc_2])
                random.shuffle(idc_rnd)

                firing_rates_ses1 = firing_rates_ses[idc_rnd < 1.0, :, :]
                firing_rates_ses2 = firing_rates_ses[idc_rnd > 0.0, :, :]
                labels_ses1 = labels_ses[idc_rnd < 1]
                labels_ses2 = labels_ses[idc_rnd > 0]

                firing_rates_ses_all.append([firing_rates_ses1, firing_rates_ses2])
                labels_ses_all.append([labels_ses1, labels_ses2])

            betas_split = []
            for split in range(n_splits):
                betas_run = []
                for run in range(2):

                    firing_mean = np.mean(firing_rates_ses_all[split][run], axis=-1, keepdims=True)
                    labels_ses_split = labels_ses_all[split][run]

                    if normalisation == 'zscore':
                        firing_mean = (firing_mean - np.mean(firing_mean, axis=0, keepdims=True)) / (
                                np.std(firing_mean, axis=0,
                                       keepdims=True) + 1)

                    elif normalisation == 'mean_centred':
                        firing_mean = firing_mean - np.mean(firing_mean, axis=0, keepdims=True)

                    elif normalisation == 'none':
                        firing_mean = firing_mean

                    elif normalisation == 'soft':
                        firing_mean = firing_mean / (
                                (np.max(firing_mean, axis=0, keepdims=True) - np.min(firing_mean, axis=0,

                                                                                     keepdims=True)) + 5)

                    labels_cue = np.array(assign_lables(labels_ses_split, condition_labels[0]))
                    labels_target = np.array(assign_lables(labels_ses_split, condition_labels[1]))
                    labels_int1 = np.array(assign_lables(labels_ses_split, condition_labels[2]))
                    labels_cue2 = np.array(assign_lables(labels_ses_split, condition_labels[3]))
                    labels_target2 = np.array(assign_lables(labels_ses_split, condition_labels[4]))
                    labels_int2 = np.array(assign_lables(labels_ses_split, condition_labels[5]))

                    if add_constant:
                        constant = np.ones_like(labels_cue).astype(float)
                        design_matrix = np.vstack(
                            [constant, labels_cue, labels_target, labels_int1, labels_cue2, labels_target2,
                             labels_int2]).T
                    else:
                        design_matrix = np.vstack(
                            [labels_cue, labels_cue, labels_target, labels_int1, labels_cue2, labels_target2,
                             labels_int2]).T

                    design_matrix = design_matrix.astype(float)

                    n_neurons = firing_mean.shape[1]
                    n_times = firing_mean.shape[-1]
                    n_models = design_matrix.shape[1]

                    design_matrix_stacked = np.concatenate([design_matrix] * n_times, axis=0)
                    firing_rates_ses_stacked = np.concatenate(np.array_split(firing_mean, n_times, axis=-1), axis=0)

                    betas_ses = np.zeros((n_neurons, n_models, 1))
                    for cell in range(n_neurons):
                        betas_ses[cell, :, :] = \
                            sp.linalg.lstsq(design_matrix_stacked, firing_rates_ses_stacked[:, cell, :])[0][:, :]
                    betas_run.append(betas_ses)
                betas_split.append(np.mean(np.array(betas_run), axis=0)[:, :, 0])
            betas_ses_all.append(np.mean(np.array(betas_split), axis=0))
        betas_part.append(np.concatenate(betas_ses_all, axis=0))

    epochs_rel = []
    for p in range(n_parts):
        if add_constant:
            epoch = betas_part[p][:, 1:4]
        else:
            epoch = betas_part[p][:, :3]
        epoch = np.concatenate([epoch[:, :, None], epoch[:, :, None]], axis=-1)
        epoch -= np.mean(epoch, axis=0, keepdims=True)
        epochs_rel.append(epoch)

    epochs_irrel = []
    for p in range(n_parts):
        if add_constant:
            epoch = betas_part[p][:, 4:]
        else:
            epoch = betas_part[p][:, 3:]

        epoch = np.concatenate([epoch[:, :, None], epoch[:, :, None]], axis=-1)
        epoch -= np.mean(epoch, axis=0, keepdims=True)
        epochs_irrel.append(epoch)

    return epochs_rel, epochs_irrel


def plot_pvals_2(data, ax, n_bootstraps, tail=1, offset=0.05, bon_correction=False, side=1,
                 metric='KL divergance estimate'):
    df_data = data[data['distribution'] == 'observed']
    n_epochs = len(np.unique(df_data['learning epoch'].values))
    rel = df_data[df_data['model'] == 'cue + shape'][metric].values
    rel = np.resize(rel, (n_epochs, n_bootstraps))
    rel_avg = np.mean(rel, axis=-1, keepdims=True)

    irrel = df_data[df_data['model'] == 'cue + width'][metric].values
    irrel = np.resize(irrel, (n_epochs, n_bootstraps))

    if side == 1:
        side_fac = 1
    elif side == 2:
        side_fac = 2

    y_avgs = rel_avg[:, 0]
    y_stds = rel.std(-1)
    if tail == 1:
        p = side_fac * (np.sum(irrel >= rel_avg, axis=-1) / n_bootstraps)
        y_pos = y_avgs + y_stds + offset
    elif tail == -1:
        p = side_fac * (np.sum(irrel <= rel_avg, axis=-1) / n_bootstraps)
        y_pos = y_avgs - y_stds - (offset * 12)

    if bon_correction:
        p = p * n_epochs

    for e in range(n_epochs):
        if p[e] > 0.05:
            star = 'ns'
            size = 12
        elif (p[e] <= 0.05) & (p[e] > 0.01):
            star = '*'
            size = 20
        elif (p[e] <= 0.01) & (p[e] > 0.001):
            star = '**'
            size = 20
        elif p[e] <= 0.001:
            star = '***'
            size = 20
        ax.text(e, y_pos[e], star, ha='center', size=size)

    return


def plot_pvals_3(y, x, p, ax, col, offset=0.05):
    if p > 0.05:
        star = 'ns'
        size = 10
    elif (p <= 0.05) & (p > 0.01):
        star = '*'
        size = 15
    elif (p <= 0.01) & (p > 0.001):
        star = '**'
        size = 15
    elif p <= 0.001:
        star = '***'
        size = 15
    ax.text(x, y + offset, star, ha='center', size=size, color=col)

    return


def compare_r2s(data1, data2, n_bootstraps, tail=1, mode='diff'):
    n_units_1 = data1.shape[0]
    n_units_2 = data2.shape[0]
    data_all = np.concatenate([data1, data2], axis=0)

    fit_diff_rnd = np.zeros((n_bootstraps))

    fit_1_obs = fit_lines(data1[:, :, 0], data1[:, :, 1], if_xgen=True)
    fit_2_obs = fit_lines(data2[:, :, 0], data2[:, :, 1], if_xgen=True)
    if mode == 'diff':
        fit_diff1 = fit_1_obs[0] - fit_1_obs[1]
        fit_diff2 = fit_2_obs[0] - fit_2_obs[1]
    elif mode == 'best':
        fit_diff1 = fit_1_obs[0]
        fit_diff2 = fit_2_obs[0]
    elif mode == 'optimal':
        fit_diff1 = fit_1_obs[1]
        fit_diff2 = fit_2_obs[1]

    fit_diff = fit_diff1 - fit_diff2

    for n_bootstrap in range(n_bootstraps):
        rnd_idc1 = np.zeros(n_units_1)
        rnd_idc2 = np.ones(n_units_2)
        rnd_idx = np.concatenate([rnd_idc1, rnd_idc2])
        random.shuffle(rnd_idx)
        data1_rnd = data_all[rnd_idx < 1, :, :]
        data2_rnd = data_all[rnd_idx > 0, :, :]

        fit_1_rnd = fit_lines(data1_rnd[:, :, 0], data1_rnd[:, :, 1], if_xgen=True)
        fit_2_rnd = fit_lines(data2_rnd[:, :, 0], data2_rnd[:, :, 1], if_xgen=True)
        if mode == 'diff':
            fit_diff1_rnd = fit_1_rnd[0] - fit_1_rnd[1]
            fit_diff2_rnd = fit_2_rnd[0] - fit_2_rnd[1]
        elif mode == 'best':
            fit_diff1_rnd = fit_1_rnd[0]
            fit_diff2_rnd = fit_2_rnd[0]
        elif mode == 'optimal':
            fit_diff1_rnd = fit_1_rnd[1]
            fit_diff2_rnd = fit_2_rnd[1]

        fit_diff_rnd[n_bootstrap] = fit_diff1_rnd - fit_diff2_rnd

    if tail == -1:
        p = (np.sum(fit_diff_rnd >= fit_diff) / n_bootstraps)
    elif tail == 1:
        p = (np.sum(fit_diff_rnd <= fit_diff) / n_bootstraps)

    return p


def data_handle(data, labels, min_n, do_rand=False):
    conditions = np.unique(labels)
    conditions_ = []
    for _ in range(len(conditions)):
        condition_dat = data[labels == conditions[_], :, :]
        if do_rand:
            idx_re = np.random.choice(np.array(list(range(0, condition_dat.shape[0]))), min_n, replace=False)
            conditions_.append(condition_dat[idx_re, :, :])
        else:
            conditions_.append(condition_dat[:min_n, :, :])
    conditions_ = np.array(conditions_)

    return np.concatenate(conditions_, axis=0)


def euclidean_distance(data1, data2):
    return np.sqrt(np.sum((np.cov(data2.T) - np.cov(data1.T)) ** 2))


def prepare_data(data, labels, which_trl='beginning', set_min_trl=None):
    n_trls_ses = []
    for sess in range(len(data)):
        n_trls_ses.append(data[sess].shape[0])
    n_trls = np.min(n_trls_ses)

    n_condi_min = []
    for sess in range(len(data)):
        if which_trl == 'beginning':
            freqs = get_freqs(labels[sess][:n_trls])
        elif which_trl == 'end':
            freqs = get_freqs(labels[sess][-n_trls:])
        elif which_trl == 'rnd':
            freqs = get_freqs(labels[sess])
        elif which_trl == 'middle':
            labels_sess = labels[sess]
            idc_half = int(len(labels_sess) / 2)
            # check whether even or odd number of trials
            if (len(labels_sess) % 2) > 0:
                labels_sess = labels_sess[idc_half - int(n_trls / 2):idc_half + int(n_trls / 2) + 1]
            else:
                labels_sess = labels_sess[idc_half - int(n_trls / 2):idc_half + int(n_trls / 2)]
            freqs = get_freqs(labels_sess)
        n_condi_min.append(freqs[min(freqs, key=freqs.get)])
    n_trl_condi = np.min(n_condi_min)

    if set_min_trl:
        n_trl_condi = set_min_trl

    dat_combined = []
    for sess in range(len(data)):
        if which_trl == 'beginning':
            data_sess = data[sess][:n_trls, :, :]
            labels_sess = labels[sess][:n_trls]
            dat_combined.append(data_handle(data_sess, labels_sess, n_trl_condi))
        elif which_trl == 'end':
            data_sess = data[sess][-n_trls:, :, :]
            labels_sess = labels[sess][-n_trls:]
            dat_combined.append(data_handle(data_sess, labels_sess, n_trl_condi))
        elif which_trl == 'rnd':
            data_sess = data_handle(data[sess], labels[sess], n_trl_condi, do_rand=False)
            dat_combined.append(data_sess)
        elif which_trl == 'middle':
            labels_sess = labels[sess]
            idc_half = int(len(labels_sess) / 2)
            # check whether even or odd number of trials
            if (len(labels_sess) % 2) > 0:
                labels_sess = labels_sess[idc_half - int(n_trls / 2):idc_half + int(n_trls / 2) + 1]
                data_sess = data[sess][idc_half - int(n_trls / 2):idc_half + int(n_trls / 2) + 1, :, :]
            elif (len(labels_sess) % 2) == 0:
                labels_sess = labels_sess[idc_half - int(n_trls / 2):idc_half + int(n_trls / 2)]
                data_sess = data[sess][idc_half - int(n_trls / 2):idc_half + int(n_trls / 2), :, :]

            dat_combined.append(data_handle(data_sess, labels_sess, n_trl_condi))
    dat_combined = np.concatenate(dat_combined, axis=1)

    labels_combined = np.sort(list(range(0, len(np.unique(labels[0])))) * n_trl_condi)

    return dat_combined, labels_combined


def get_shattering_ids(cue, target, width, reward):
    n_condi = len(cue)

    all_combos = list(itertools.combinations(list(range(n_condi)), int(n_condi / 2)))
    n_combos = int(len(all_combos) / 2)

    decoding_targets = np.zeros((n_combos, n_condi))
    cue_tgt_width_reward_ids = np.zeros(int(n_condi / 2))
    for i in range(n_combos):
        decoding_targets[i, all_combos[i]] = 1

        # Find cue ids
        if np.sum(np.abs(decoding_targets[i, :] - cue)) == 0 or np.sum(np.abs(decoding_targets[i, :] - (1 - cue))) == 0:
            cue_tgt_width_reward_ids[0] = i

        # Find target ids
        if np.sum(np.abs(decoding_targets[i, :] - target)) == 0 or np.sum(
                np.abs(decoding_targets[i, :] - (1 - target))) == 0:
            cue_tgt_width_reward_ids[1] = i

        # Find width ids
        if np.sum(np.abs(decoding_targets[i, :] - width)) == 0 or np.sum(
                np.abs(decoding_targets[i, :] - (1 - width))) == 0:
            cue_tgt_width_reward_ids[2] = i

        # Find reward ids
        if np.sum(np.abs(decoding_targets[i, :] - reward)) == 0 or np.sum(
                np.abs(decoding_targets[i, :] - (1 - reward))) == 0:
            cue_tgt_width_reward_ids[3] = i

    cross_gen_decoding_train_ids = []
    cross_gen_decoding_test_ids = []
    for i in range(n_combos):
        current_zeros = np.squeeze(np.where(decoding_targets[i, :] == 0))
        current_combs1 = list(itertools.combinations(current_zeros.tolist(), 2))
        current_ones = np.squeeze(np.where(decoding_targets[i, :] == 1))
        current_combs2 = list(itertools.combinations(current_ones.tolist(), 2))

        n_combs = len(current_combs1)
        current_train = []
        current_test = []
        for j in range(n_combs):
            for k in range(n_combs):
                current_train.append([current_combs1[j], current_combs2[k]])

                current_test_zeros = list(np.setdiff1d(current_zeros, np.array(current_combs1[j])))
                current_test_ones = list(np.setdiff1d(current_ones, np.array(current_combs2[k])))
                current_test.append([current_test_zeros, current_test_ones])

        cross_gen_decoding_train_ids.append(current_train)
        cross_gen_decoding_test_ids.append(current_test)

    return decoding_targets, cue_tgt_width_reward_ids, cross_gen_decoding_train_ids, cross_gen_decoding_test_ids


def decode(X, y, method='svm', n_inter=5, return_inter=False, n_jobs=None):
    y = np.array(y)

    scores_all = []
    for n in range(n_inter):
        if method == 'svm':
            clf1 = make_pipeline(StandardScaler(), SVM(C=5e-4))
            clf2 = make_pipeline(StandardScaler(), SVM(C=5e-4))
            if n_jobs != None:
                clf1 = SlidingEstimator(clf1, verbose='warning', n_jobs=n_jobs)
                clf2 = SlidingEstimator(clf2, verbose='warning', n_jobs=n_jobs)

        elif method == 'lda':
            clf1 = make_pipeline(StandardScaler(), LDA())
            if n_jobs != None:
                clf1 = SlidingEstimator(clf1, verbose='warning', n_jobs=n_jobs)
            clf2 = make_pipeline(StandardScaler(), LDA())
            if n_jobs != None:
                clf2 = SlidingEstimator(clf2, verbose='warning', n_jobs=n_jobs)

        n_trls = X.shape[0]
        idx_rnd = np.concatenate([np.zeros(n_trls // 2), np.ones(n_trls // 2)])
        if (n_trls % 2) > 0:
            idx_rnd = np.concatenate([idx_rnd, [1.0]])
        random.shuffle(idx_rnd)

        if n_jobs != None:
            X1 = X[idx_rnd < 1, :, :]
            y1 = y[idx_rnd < 1]
            X2 = X[idx_rnd > 0, :, :]
            y2 = y[idx_rnd > 0]
        else:
            X1 = X[idx_rnd < 1, :]
            y1 = y[idx_rnd < 1]
            X2 = X[idx_rnd > 0, :]
            y2 = y[idx_rnd > 0]

        clf1.fit(X1, y1)
        score1 = clf1.score(X2, y2)

        clf2.fit(X2, y2)
        score2 = clf2.score(X1, y1)
        scores_all.append(np.array([score1, score2]).mean(0))
    if return_inter:
        return scores_all
    else:
        return np.mean(scores_all)


def get_decoding(data, labels, variables, time_window, method, n_jobs=None):
    decoding_targets, cue_tgt_width_reward_ids, _, _ = get_shattering_ids(variables[0], variables[1], variables[2],
                                                                          variables[3])

    n_combos = decoding_targets.shape[0]
    scores = []
    print('Cutting along all possible axes (shattering dimensionality)')
    for combo in tqdm(range(n_combos)):
        if n_jobs != None:
            X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=True)
        else:
            X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=False)
        y = assign_lables(labels, decoding_targets[combo, :])
        scores.append(decode(X, y, method=method, n_jobs=n_jobs))

    score_cue = scores[int(cue_tgt_width_reward_ids[0])]
    score_target = scores[int(cue_tgt_width_reward_ids[1])]
    score_width = scores[int(cue_tgt_width_reward_ids[2])]
    score_rew = scores[int(cue_tgt_width_reward_ids[3])]
    score_shattering = scores

    var_ids = [int(cue_tgt_width_reward_ids) for cue_tgt_width_reward_ids in cue_tgt_width_reward_ids]
    scores_irrel = np.delete(scores, obj=var_ids)

    return [score_cue, score_target, score_width, score_rew, score_shattering], scores_irrel


def get_decoding_null(data, labels, time_window, method, n_reps=100, n_jobs=None):
    if len(np.unique(labels)) < 9:
        fac = [0, 0, 0, 0, 1, 1, 1, 1]
    else:
        fac = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    print('Generating the null decoding distribution...')
    scores = []
    for rep in tqdm(range(n_reps)):
        if n_jobs != None:
            X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=True)
        else:
            X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=False)
        y = assign_lables(labels, factor=fac).copy()
        if len(np.unique(labels)) < 3:
            y = labels.copy()
        random.shuffle(y)
        scores.append(decode(X, y, method=method, n_jobs=n_jobs))

    return np.array(scores)


def decode_xgen(X1, X2, y1, y2, method='svm', n_jobs=-1):
    # prepare a series of classifier applied at each time sample
    if method == 'svm':
        clf1 = make_pipeline(StandardScaler(), SVM(C=5e-4))
        if n_jobs != None:
            clf1 = SlidingEstimator(clf1, verbose=False, n_jobs=-1)
        clf2 = make_pipeline(StandardScaler(), SVM(C=5e-4))
        if n_jobs != None:
            clf2 = SlidingEstimator(clf2, verbose=False, n_jobs=-1)
    elif method == 'lda':
        clf1 = make_pipeline(StandardScaler(), LDA())
        if n_jobs != None:
            clf1 = SlidingEstimator(clf1, verbose=False, n_jobs=-1)
        clf2 = make_pipeline(StandardScaler(), LDA())
        if n_jobs != None:
            clf2 = SlidingEstimator(clf2, verbose=False, n_jobs=-1)

    clf1.fit(X1, y1)
    score1 = clf1.score(X2, y2)

    clf2.fit(X2, y2)
    score2 = clf2.score(X1, y1)

    return np.array([score1, score2]).mean(0)


def get_x_gen_null(data, labels, method, n_reps=100, n_jobs=None):
    if len(np.unique(labels)) < 9:
        fac = [0, 0, 0, 0, 1, 1, 1, 1]
    else:
        fac = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    print('Generating the null x-gen distribution...')
    scores = []
    for rep in tqdm(range(n_reps)):
        X = data
        y = assign_lables(labels, factor=fac).copy()
        if n_jobs != None:
            X1 = X[::2, :, :]
            X2 = X[1::2, :, :]
        else:
            X1 = X[::2, :]
            X2 = X[1::2, :]
        y1 = y[::2]
        y2 = y[1::2]

        random.shuffle(y1)
        random.shuffle(y2)
        scores.append(decode_xgen(X1, X2, y1, y2, method=method, n_jobs=n_jobs))

    return np.array(scores)


def get_xgen(data, labels, variables, time_window, method='svm', mode='full', n_jobs=None, verbose=False):
    decoding_targets, cue_tgt_width_reward_ids, cross_gen_decoding_train_ids, cross_gen_decoding_test_ids = get_shattering_ids(
        variables[0], variables[1], variables[2], variables[3])

    var_ids = [int(cue_tgt_width_reward_ids) for cue_tgt_width_reward_ids in cue_tgt_width_reward_ids]
    n_combos = len(cross_gen_decoding_train_ids)

    if mode == 'only_rel':
        combo_list = var_ids
    elif mode == 'full':
        combo_list = list(range(n_combos))

    scores = []
    if verbose:
        print('Cutting along all possible axes (xgen)')
    for combo in tqdm(combo_list, disable=not verbose):
        xgen_train = cross_gen_decoding_train_ids[combo]
        xgen_test = cross_gen_decoding_test_ids[combo]

        if n_jobs != None:
            X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=True)
        else:
            X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=False)

        n_axes = len(xgen_train)
        scores_axes = []
        for axis in range(n_axes):
            y1 = np.array(xgen_train[axis]).flatten()
            X1 = []
            for _ in range(len(y1)):
                if n_jobs != None:
                    X1.append(X[labels == y1[_], :, :])
                else:
                    X1.append(X[labels == y1[_], :])
            X1 = np.concatenate(X1, axis=0)

            y2 = np.array(xgen_test[axis]).flatten()
            X2 = []
            for _ in range(len(y1)):
                if n_jobs != None:
                    X2.append(X[labels == y2[_], :, :])
                else:
                    X2.append(X[labels == y2[_], :])
            X2 = np.concatenate(X2, axis=0)

            y = np.concatenate([np.zeros(int(X1.shape[0] / 2)), np.ones(int(X1.shape[0] / 2))])
            scores_axes.append(decode_xgen(X1, X2, y, y, method=method, n_jobs=n_jobs))
        scores.append(np.mean(scores_axes))

    if mode == 'full':
        scores_rel = [scores[var_ids[0]], scores[var_ids[1]], scores[var_ids[2]], scores[var_ids[3]]]
        scores_irrel = np.delete(scores, obj=var_ids)

        return scores_rel, scores_irrel

    elif mode == 'only_rel':
        return np.array(scores)


def get_decoding_models(n_neurons=400, noise_std=1, n_trials=20, seed=42):
    np.random.seed(seed)
    conds = np.ones((4, 3))
    conds[0, :2] = -1
    conds[1, [0, 2]] = -1
    conds[2, 1:] = -1
    conds = np.array([conds] * n_trials)

    clfc = SVM(C=1e-5)
    clfs = SVM(C=1e-5)
    clfr = SVM(C=1e-5)

    N_std = [[n_neurons, noise_std]]  # number of neurons and std of noise
    targets_r = [1, -1, -1, 1]
    targets_s = [-1, 1, -1, 1]
    targets_c = [-1, -1, 1, 1]
    decoding_tot = np.zeros((len(N_std), 100, 3, 2))
    cross_decoding_tot = np.zeros((len(N_std), 100, 3, 2))
    for counter, n_std in enumerate(N_std):
        N = n_std[0]
        noise_std = n_std[1]
        for model in range(100):
            for rand_opt in range(2):
                if rand_opt == 0:
                    betas = np.random.normal(0, 1, (3, N))
                else:
                    cov = np.diag([0, 0, 3])
                    betas = np.random.multivariate_normal(np.zeros(3), cov, N).T

                x_train0 = conds @ betas + noise_std * np.random.normal(0, 1, (n_trials, 4, N))
                x_test0 = conds @ betas + noise_std * np.random.normal(0, 1, (n_trials, 4, N))

                x_train0_long = np.concatenate(np.array_split(x_train0, n_trials, axis=0), axis=1)[0, :, :]
                x_test0_long = np.concatenate(np.array_split(x_test0, n_trials, axis=0), axis=1)[0, :, :]

                clfc.fit(x_train0_long, np.concatenate([targets_c] * n_trials))
                clfs.fit(x_train0_long, np.concatenate([targets_s] * n_trials))
                clfr.fit(x_train0_long, np.concatenate([targets_r] * n_trials))

                decoding_tot[counter, model, 0, rand_opt] = clfc.score(x_test0_long,
                                                                       np.concatenate([targets_c] * n_trials))
                decoding_tot[counter, model, 1, rand_opt] = clfs.score(x_test0_long,
                                                                       np.concatenate([targets_s] * n_trials))
                decoding_tot[counter, model, 2, rand_opt] = clfr.score(x_test0_long,
                                                                       np.concatenate([targets_r] * n_trials))

                # Cross decoding
                r_train_id = [[0, 1], [0, 2], [3, 1], [3, 2]]
                r_test_id = [[3, 2], [3, 1], [0, 2], [0, 1]]
                s_train_id = [[0, 1], [0, 3], [2, 1], [2, 3]]
                s_test_id = [[2, 3], [2, 1], [0, 3], [0, 1]]
                c_train_id = [[0, 2], [0, 3], [1, 2], [1, 3]]
                c_test_id = [[1, 3], [1, 2], [0, 3], [0, 2]]
                for i in range(4):
                    # color
                    x_train1 = x_train0[:, c_train_id[i], :]
                    x_test1 = x_test0[:, c_test_id[i], :]
                    x_train1 = np.concatenate(np.array_split(x_train1, n_trials, axis=0), axis=1)[0, :, :]
                    x_test1 = np.concatenate(np.array_split(x_test1, n_trials, axis=0), axis=1)[0, :, :]
                    clfc.fit(x_train1, [0, 1] * n_trials)
                    cross_decoding_tot[counter, model, 0, rand_opt] += 0.25 * clfc.score(x_test1, [0, 1] * n_trials)

                    # shape
                    x_train1 = x_train0[:, s_train_id[i], :]
                    x_test1 = x_test0[:, s_test_id[i], :]
                    x_train1 = np.concatenate(np.array_split(x_train1, n_trials, axis=0), axis=1)[0, :, :]
                    x_test1 = np.concatenate(np.array_split(x_test1, n_trials, axis=0), axis=1)[0, :, :]
                    clfs.fit(x_train1, [0, 1] * n_trials)
                    cross_decoding_tot[counter, model, 1, rand_opt] += 0.25 * clfs.score(x_test1, [0, 1] * n_trials)

                    # reward
                    x_train1 = x_train0[:, r_train_id[i], :]
                    x_test1 = x_test0[:, r_test_id[i], :]
                    x_train1 = np.concatenate(np.array_split(x_train1, n_trials, axis=0), axis=1)[0, :, :]
                    x_test1 = np.concatenate(np.array_split(x_test1, n_trials, axis=0), axis=1)[0, :, :]
                    clfr.fit(x_train1, [0, 1] * n_trials)
                    cross_decoding_tot[counter, model, 2, rand_opt] += 0.25 * clfr.score(x_test1, [0, 1] * n_trials)

    m_decode = np.mean(decoding_tot[0, :], axis=0)
    m_cross_gen = np.mean(cross_decoding_tot[0, :], axis=0)

    return m_decode, m_cross_gen


def get_decoding_models(n_neurons=400, noise_std=1, n_trials=20, seed=42):
    np.random.seed(seed)
    conds = np.ones((4, 3))
    conds[0, :2] = -1
    conds[1, [0, 2]] = -1
    conds[2, 1:] = -1
    conds = np.array([conds] * n_trials)

    clfc = SVM(C=1e-5)
    clfs = SVM(C=1e-5)
    clfr = SVM(C=1e-5)

    N_std = [[n_neurons, noise_std]]  # number of neurons and std of noise
    targets_r = [1, -1, -1, 1]
    targets_s = [-1, 1, -1, 1]
    targets_c = [-1, -1, 1, 1]
    decoding_tot = np.zeros((len(N_std), 100, 3, 2))
    cross_decoding_tot = np.zeros((len(N_std), 100, 3, 2))
    for counter, n_std in enumerate(N_std):
        N = n_std[0]
        noise_std = n_std[1]
        for model in range(100):
            for rand_opt in range(2):
                if rand_opt == 0:
                    betas = np.random.normal(0, 1, (3, N))
                else:
                    cov = np.diag([0, 0, 3])
                    betas = np.random.multivariate_normal(np.zeros(3), cov, N).T

                x_train0 = conds @ betas + noise_std * np.random.normal(0, 1, (n_trials, 4, N))
                x_test0 = conds @ betas + noise_std * np.random.normal(0, 1, (n_trials, 4, N))

                x_train0_long = np.concatenate(np.array_split(x_train0, n_trials, axis=0), axis=1)[0, :, :]
                x_test0_long = np.concatenate(np.array_split(x_test0, n_trials, axis=0), axis=1)[0, :, :]

                clfc.fit(x_train0_long, np.concatenate([targets_c] * n_trials))
                clfs.fit(x_train0_long, np.concatenate([targets_s] * n_trials))
                clfr.fit(x_train0_long, np.concatenate([targets_r] * n_trials))

                decoding_tot[counter, model, 0, rand_opt] = clfc.score(x_test0_long,
                                                                       np.concatenate([targets_c] * n_trials))
                decoding_tot[counter, model, 1, rand_opt] = clfs.score(x_test0_long,
                                                                       np.concatenate([targets_s] * n_trials))
                decoding_tot[counter, model, 2, rand_opt] = clfr.score(x_test0_long,
                                                                       np.concatenate([targets_r] * n_trials))

                # Cross decoding
                r_train_id = [[0, 1], [0, 2], [3, 1], [3, 2]]
                r_test_id = [[3, 2], [3, 1], [0, 2], [0, 1]]
                s_train_id = [[0, 1], [0, 3], [2, 1], [2, 3]]
                s_test_id = [[2, 3], [2, 1], [0, 3], [0, 1]]
                c_train_id = [[0, 2], [0, 3], [1, 2], [1, 3]]
                c_test_id = [[1, 3], [1, 2], [0, 3], [0, 2]]
                for i in range(4):
                    # color
                    x_train1 = x_train0[:, c_train_id[i], :]
                    x_test1 = x_test0[:, c_test_id[i], :]
                    x_train1 = np.concatenate(np.array_split(x_train1, n_trials, axis=0), axis=1)[0, :, :]
                    x_test1 = np.concatenate(np.array_split(x_test1, n_trials, axis=0), axis=1)[0, :, :]
                    clfc.fit(x_train1, [0, 1] * n_trials)
                    cross_decoding_tot[counter, model, 0, rand_opt] += 0.25 * clfc.score(x_test1, [0, 1] * n_trials)

                    # shape
                    x_train1 = x_train0[:, s_train_id[i], :]
                    x_test1 = x_test0[:, s_test_id[i], :]
                    x_train1 = np.concatenate(np.array_split(x_train1, n_trials, axis=0), axis=1)[0, :, :]
                    x_test1 = np.concatenate(np.array_split(x_test1, n_trials, axis=0), axis=1)[0, :, :]
                    clfs.fit(x_train1, [0, 1] * n_trials)
                    cross_decoding_tot[counter, model, 1, rand_opt] += 0.25 * clfs.score(x_test1, [0, 1] * n_trials)

                    # reward
                    x_train1 = x_train0[:, r_train_id[i], :]
                    x_test1 = x_test0[:, r_test_id[i], :]
                    x_train1 = np.concatenate(np.array_split(x_train1, n_trials, axis=0), axis=1)[0, :, :]
                    x_test1 = np.concatenate(np.array_split(x_test1, n_trials, axis=0), axis=1)[0, :, :]
                    clfr.fit(x_train1, [0, 1] * n_trials)
                    cross_decoding_tot[counter, model, 2, rand_opt] += 0.25 * clfr.score(x_test1, [0, 1] * n_trials)

    m_decode = np.mean(decoding_tot[0, :], axis=0)
    m_cross_gen = np.mean(cross_decoding_tot[0, :], axis=0)

    return m_decode, m_cross_gen


def pearsonr(X, Y, axis=None, keepdims=False, strip_nans=False):
    """
    Pearson correlation across a specific axis.
    """

    if strip_nans:
        mymean = np.nanmean
        mystd = np.nanstd
        mysum = np.nansum
    else:
        mymean = np.mean
        mystd = np.std
        mysum = np.sum

    should_squeeze = axis is not None and not keepdims
    if axis is None:
        X = X.ravel()
        Y = Y.ravel()
        axis = 0

    xbar = mymean(X, axis=axis, keepdims=True)
    ybar = mymean(Y, axis=axis, keepdims=True)
    ssx = mysum((X - xbar) ** 2, axis=axis, keepdims=True)
    ssy = mysum((Y - ybar) ** 2, axis=axis, keepdims=True)

    # the following else-block is equivalent to:
    # num = np.sum( (X-xbar)*(Y-ybar) , axis=axis, keepdims=True)
    # but use MUCH less memory as they accumulate the sum in a loop.
    # the two approaches use the same amount of cputime

    if strip_nans:
        # use the memory-inefficient way in case we have to deal with nans
        num = mysum((X - xbar) * (Y - ybar), axis=axis, keepdims=True)
    else:
        tmpX = X.take(0, axis=axis).reshape(xbar.shape)
        tmpY = Y.take(0, axis=axis).reshape(ybar.shape)
        num = (tmpX - xbar) * (tmpY - ybar)
        for k in range(1, X.shape[axis]):
            tmpX = X.take(k, axis=axis).reshape(xbar.shape)
            tmpY = Y.take(k, axis=axis).reshape(ybar.shape)
            num += (tmpX - xbar) * (tmpY - ybar)

    denom = np.sqrt(ssx) * np.sqrt(ssy)
    r = num / denom

    if should_squeeze:
        s = list(r.shape)
        assert (s[axis] == 1)  # the axis dimension should now be singleton
        s.pop(axis)  # so remove it
        r = r.reshape(s)

    return r


def dist_random_data2(const_coeffs, metric='euclidean distance', rnd_model='gaussian (spherical)', n_bootstraps=1000,
                      relative_dist=True, bon_correction=False):
    n_epochs = len(const_coeffs)
    n_pairs = const_coeffs[0].shape[0]
    dist_data = np.zeros((n_epochs, n_pairs, n_bootstraps))
    dist_rnd = np.zeros((n_epochs, n_pairs, n_bootstraps))
    dist_str = np.zeros((n_epochs, n_pairs, n_bootstraps))
    for part in range(n_epochs):
        for n_bootstrap in tqdm(range(n_bootstraps)):
            for pair in range(n_pairs):
                data_coeffs = const_coeffs[part][pair, :, :]

                opt_cov = np.diag([0, 0, np.cov(data_coeffs.T)[2, 2]])
                s_opt = np.random.multivariate_normal(np.zeros(data_coeffs.shape[1]),
                                                      opt_cov,
                                                      data_coeffs.shape[0])

                m = np.mean(np.diag(np.cov(data_coeffs.T)))
                s_rnd_1 = np.random.multivariate_normal(np.zeros(data_coeffs.shape[1]),
                                                        np.diag([m, m, m]),
                                                        data_coeffs.shape[0])

                s_rnd_2 = np.random.multivariate_normal(np.zeros(data_coeffs.shape[1]),
                                                        np.diag([m, m, m]),
                                                        data_coeffs.shape[0])

                if metric == 'euclidean distance':
                    dist_data[part, pair, n_bootstrap] = euclidean_distance(s_rnd_1, data_coeffs)
                    dist_rnd[part, pair, n_bootstrap] = euclidean_distance(s_rnd_1, s_rnd_2)
                    dist_str[part, pair, n_bootstrap] = euclidean_distance(s_rnd_1, s_opt)
                elif metric == 'KL divergance estimate':
                    dist_data[part, pair, n_bootstrap] = 0.5 * (
                            KLdivergence(s_rnd_1, data_coeffs) + KLdivergence(data_coeffs, s_rnd_1))
                    dist_rnd[part, pair, n_bootstrap] = 0.5 * (
                            KLdivergence(s_rnd_1, s_rnd_2) + KLdivergence(s_rnd_2, s_rnd_1))
                    dist_str[part, pair, n_bootstrap] = 0.5 * (
                            KLdivergence(s_rnd_1, s_opt) + KLdivergence(s_opt, s_rnd_1))

    dist_data = np.reshape(dist_data, (dist_data.shape[0], dist_data.shape[1] * dist_data.shape[2]))
    dist_rnd = np.reshape(dist_rnd, (dist_rnd.shape[0], dist_rnd.shape[1] * dist_rnd.shape[2]))
    dist_str = np.reshape(dist_str, (dist_str.shape[0], dist_str.shape[1] * dist_str.shape[2]))

    if relative_dist:
        dist_rnd_avg = np.mean(dist_rnd, keepdims=True, axis=-1)
        dist_str_avg = np.mean(dist_str, keepdims=True, axis=-1)
        dist_data -= dist_rnd_avg
        dist_rnd -= dist_rnd_avg
        dist_str -= dist_rnd_avg
        dist_data /= dist_str_avg
        dist_rnd /= dist_str_avg
        dist_str /= dist_str_avg

    p = 2 * (np.sum(dist_rnd >= np.mean(dist_data, axis=-1, keepdims=True), axis=-1) / n_bootstraps)

    if bon_correction:
        p = p * n_epochs
    print('p-values:')
    print(p)

    epoch = np.sort(list(range(1, n_epochs + 1)) * n_bootstraps * n_pairs)
    epoch_labels = np.concatenate([epoch, epoch, epoch])
    dist_label = np.concatenate([['observed'] * (n_bootstraps * n_epochs * n_pairs),
                                 [rnd_model] * (n_bootstraps * n_epochs * n_pairs),
                                 ['structured'] * (n_bootstraps * n_epochs * n_pairs)])
    data_df = np.reshape(dist_data, dist_data.shape[0] * dist_data.shape[1])
    rnd_df = np.reshape(dist_rnd, dist_rnd.shape[0] * dist_rnd.shape[1])
    str_df = np.reshape(dist_str, dist_str.shape[0] * dist_str.shape[1])
    all_df = np.concatenate([data_df, rnd_df, str_df])

    df = pd.DataFrame(np.array([all_df, epoch_labels, dist_label]).T,
                      columns=[metric, 'learning epoch', 'distribution'])
    df[metric] = df[metric].astype(float)
    df['divergence from'] = 'random selectivity'

    return df, p, np.array([dist_data, dist_rnd, dist_str])


def dist_structured_data2(const_coeffs, metric='euclidean distance', rnd_model='gaussian (spherical)',
                          n_bootstraps=1000, relative_dist=True, bon_correction=False):
    n_epochs = len(const_coeffs)
    n_pairs = const_coeffs[0].shape[0]
    dist_data = np.zeros((n_epochs, n_pairs, n_bootstraps))
    dist_rnd = np.zeros((n_epochs, n_pairs, n_bootstraps))
    dist_str = np.zeros((n_epochs, n_pairs, n_bootstraps))
    for part in range(n_epochs):
        for n_bootstrap in tqdm(range(n_bootstraps)):
            for pair in range(n_pairs):
                data_coeffs = const_coeffs[part][pair, :, :]

                opt_cov = np.diag([0, 0, np.cov(data_coeffs.T)[2, 2]])
                s_opt_1 = np.random.multivariate_normal(np.zeros(data_coeffs.shape[1]),
                                                        opt_cov,
                                                        data_coeffs.shape[0])

                s_opt_2 = np.random.multivariate_normal(np.zeros(data_coeffs.shape[1]),
                                                        opt_cov,
                                                        data_coeffs.shape[0])

                m = np.mean(np.diag(np.cov(data_coeffs.T)))
                s_rnd = np.random.multivariate_normal(np.zeros(data_coeffs.shape[1]),
                                                      np.diag([m, m, m]),
                                                      data_coeffs.shape[0])

                if metric == 'euclidean distance':
                    dist_data[part, pair, n_bootstrap] = euclidean_distance(s_opt_1, data_coeffs)
                    dist_rnd[part, pair, n_bootstrap] = euclidean_distance(s_opt_1, s_rnd)
                    dist_str[part, pair, n_bootstrap] = euclidean_distance(s_opt_1, s_opt_2)
                elif metric == 'KL divergance estimate':
                    dist_data[part, pair, n_bootstrap] = 0.5 * (
                            KLdivergence(s_opt_1, data_coeffs) + KLdivergence(data_coeffs, s_opt_1))
                    dist_rnd[part, pair, n_bootstrap] = 0.5 * (
                            KLdivergence(s_opt_1, s_rnd) + KLdivergence(s_rnd, s_opt_1))
                    dist_str[part, pair, n_bootstrap] = 0.5 * (
                            KLdivergence(s_opt_1, s_opt_2) + KLdivergence(s_opt_2, s_opt_1))

    dist_data = np.reshape(dist_data, (dist_data.shape[0], dist_data.shape[1] * dist_data.shape[2]))
    dist_rnd = np.reshape(dist_rnd, (dist_rnd.shape[0], dist_rnd.shape[1] * dist_rnd.shape[2]))
    dist_str = np.reshape(dist_str, (dist_str.shape[0], dist_str.shape[1] * dist_str.shape[2]))

    if relative_dist:
        dist_str_avg = np.mean(dist_str, keepdims=True, axis=-1)
        dist_rnd_avg = np.mean(dist_rnd, keepdims=True, axis=-1)
        dist_data -= dist_str_avg
        dist_rnd -= dist_str_avg
        dist_str -= dist_str_avg
        dist_data /= dist_rnd_avg
        dist_rnd /= dist_rnd_avg
        dist_str /= dist_rnd_avg

    p = 2 * (np.sum(dist_rnd <= np.mean(dist_data, axis=-1, keepdims=True), axis=-1) / n_bootstraps)

    if bon_correction:
        p = p * n_epochs
    print('p-values:')
    print(p)

    epoch = np.sort(list(range(1, n_epochs + 1)) * n_bootstraps * n_pairs)
    epoch_labels = np.concatenate([epoch, epoch, epoch])
    dist_label = np.concatenate([['observed'] * (n_bootstraps * n_epochs * n_pairs),
                                 [rnd_model] * (n_bootstraps * n_epochs * n_pairs),
                                 ['structured'] * (n_bootstraps * n_epochs * n_pairs)])
    data_df = np.reshape(dist_data, dist_data.shape[0] * dist_data.shape[1])
    rnd_df = np.reshape(dist_rnd, dist_rnd.shape[0] * dist_rnd.shape[1])
    str_df = np.reshape(dist_str, dist_str.shape[0] * dist_str.shape[1])
    all_df = np.concatenate([data_df, rnd_df, str_df])

    df = pd.DataFrame(np.array([all_df, epoch_labels, dist_label]).T,
                      columns=[metric, 'learning epoch', 'distribution'])
    df[metric] = df[metric].astype(float)
    df['divergence from'] = 'structured selectivity'

    return df, p, np.array([dist_data, dist_rnd, dist_str])


def get_decoding_exp2(data, labels, variables, time_window, method):
    n_combos = len(variables)
    scores = []
    print('Decoding task variables')
    for combo in tqdm(range(n_combos)):
        X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=True)
        y = assign_lables(labels, variables[combo])
        scores.append(decode(X, y, method=method, return_inter=True))

    return np.array(scores)[:, :, 0]


def shattering_dim_rel(data, labels, fac_new, time_window, method='svm'):
    labels_new = assign_lables(labels, factor=fac_new)
    n_condi = len(np.unique(fac_new))
    all_combos = list(itertools.combinations(list(range(n_condi)), int(n_condi / 2)))
    n_combos = int(len(all_combos) / 2)

    combo_targets = np.ones((n_combos, n_condi))
    for combo in range(n_combos):
        combo_targets[combo, all_combos[combo][0]] = 0
        combo_targets[combo, all_combos[combo][1]] = 0

    scores = []
    print('Cutting along all possible axes (shattering dimensionality)')
    for combo in tqdm(range(n_combos)):
        X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=True)
        y = assign_lables(labels_new, combo_targets[combo, :])
        scores.append(decode(X, y, method=method))

    return np.array(scores).mean()


def cross_decoding_1axis(data, labels, target_ax, splitting_ax, method='svm', n_jobs=-1, if_rnd=False):
    labels_splitting_ax = np.array(assign_lables(labels, splitting_ax))
    labels_target_ax = np.array(assign_lables(labels, target_ax))
    if if_rnd:
        random.shuffle(labels_target_ax)
    X1 = data[labels_splitting_ax < 1, :, :]
    y1 = labels_target_ax[labels_splitting_ax < 1]
    X2 = data[labels_splitting_ax > 0, :, :]
    y2 = labels_target_ax[labels_splitting_ax > 0]

    score = decode_xgen(X1, X2, y1, y2, method=method, n_jobs=n_jobs)

    return score


def get_cell_frate(dat, lab):
    cells_lis_cue = []
    cells_lis_shape = []
    cells_lis_xor = []

    cue = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    shape = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    width = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    xor = np.array([1, 1, 0, 0, 0, 0, 1, 1])

    for _ in range(len(dat)):
        lables_fac = assign_lables(lab[_], factor=cue)
        cells_lis_cue.append(condi_avg(dat[_], lables_fac))

        lables_fac = assign_lables(lab[_], factor=shape)
        cells_lis_shape.append(condi_avg(dat[_], lables_fac))

        lables_fac = assign_lables(lab[_], factor=xor)
        cells_lis_xor.append(condi_avg(dat[_], lables_fac))

    cells_arr_cue = np.concatenate(cells_lis_cue, axis=1)
    cells_lis_shape = np.concatenate(cells_lis_shape, axis=1)
    cells_lis_xor = np.concatenate(cells_lis_xor, axis=1)

    return cells_arr_cue, cells_lis_shape, cells_lis_xor


def decode_epoch(data, labels, n_reps=100, method='svm', n_inter=1, n_jobs=-1):
    obs = np.array(decode(data, labels, method=method, n_inter=n_inter, return_inter=True, n_jobs=n_jobs)).mean(0)

    rnd = np.zeros((n_reps, data.shape[-1]))
    for _ in range(n_reps):

        n_trls = data.shape[0]
        idx_rnd = np.concatenate([np.zeros(n_trls // 2), np.ones(n_trls // 2)])
        if (n_trls % 2) > 0:
            idx_rnd = np.concatenate([idx_rnd, [1.0]])
        random.shuffle(idx_rnd)

        rnd[_, :] = np.array(
            decode(data, idx_rnd, method=method, n_inter=n_inter, return_inter=True, n_jobs=n_jobs)).mean(0)

    return obs, rnd


def decode_epoch_diff(data1, data2, labels, n_reps=100, method='svm', n_inter=10, tail=1, n_jobs=-1):
    n_cells_1 = data1.shape[1]
    n_cells_2 = data2.shape[1]
    data_all = np.concatenate([data1, data2], axis=1)

    obs1 = np.array(decode(data1, labels, method=method, n_inter=n_inter, return_inter=True, n_jobs=n_jobs)).mean(0)
    obs2 = np.array(decode(data2, labels, method=method, n_inter=n_inter, return_inter=True, n_jobs=n_jobs)).mean(0)

    if tail == 1:
        obs = obs1 - obs2
    elif tail == -1:
        obs = obs2 - obs1

    rnd = np.zeros((n_reps, data1.shape[-1]))
    for _ in range(n_reps):
        cell_idx = np.concatenate([np.zeros(n_cells_1), np.ones(n_cells_2)])
        random.shuffle(cell_idx)
        data1_rnd = data_all[:, cell_idx == 0, :]
        data2_rnd = data_all[:, cell_idx == 1, :]

        rnd_1 = np.array(
            decode(data1_rnd, labels, method=method, n_inter=n_inter, return_inter=True, n_jobs=n_jobs)).mean(0)
        rnd_2 = np.array(
            decode(data2_rnd, labels, method=method, n_inter=n_inter, return_inter=True, n_jobs=n_jobs)).mean(0)

        if tail == 1:
            rnd[_, :] = rnd_1 - rnd_2
        elif tail == -1:
            rnd[_, :] = rnd_2 - rnd_1

    return obs, rnd


def smooth(x, window_len=5, window='hanning'):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2):-int((window_len / 2))]


def permutation_test(obsdat, rnddat, clustercorrect=True, clusteralpha=0.05, tail=0):
    """
    Performs an (optionally cluster-corrected) permutation test of the observed
    data, given the pre-computed randomizations. rnddat must have one extra
    trailing dimension compared to obsdat.
    """
    if tail == 0:
        alpha_2tail = clusteralpha / 2

        clusterThreshold_right = np.percentile(rnddat, 100 * (1 - alpha_2tail), axis=obsdat.ndim)
        clusterThreshold_left = np.percentile(rnddat, 100 * alpha_2tail, axis=obsdat.ndim)

    elif tail == -1:
        clusterThreshold = np.percentile(rnddat, 100 * clusteralpha, axis=obsdat.ndim)

    elif tail == 1:
        clusterThreshold = np.percentile(rnddat, 100 * (1 - clusteralpha), axis=obsdat.ndim)

    p = np.ones_like(obsdat, dtype='float64')

    # uncorrected 'test'
    if not clustercorrect:
        for inds, value in np.ndenumerate(obsdat):
            rnddat[inds].sort()
            p[inds] = 1 - np.searchsorted(rnddat[inds], value) / rnddat.shape[-1]
        return p

    # subfunction to compute clusterstats in one dataset (observed/randomized)
    def compute_clusterstats(dat, getinds=True):

        if tail == 0:
            clusterCandidates_right = dat > clusterThreshold_right
            clusterCandidates_left = dat < clusterThreshold_left
            clusterCandidates = np.logical_or(clusterCandidates_right, clusterCandidates_left)

        elif tail == -1:
            clusterCandidates = dat < clusterThreshold
        elif tail == 1:
            clusterCandidates = dat > clusterThreshold

        # label connected tiles
        labelled, numfeat = label(clusterCandidates, output='uint32')

        # compute aggregate cluster statistic for each cluster
        # this is a quick way to use cluster size as the clusterstat.
        # for other clusterstats (e.g. summed stat) a few more lines are needed
        clusterNums, clusterStats = np.unique(labelled.ravel(), return_counts=True)

        # remove 0, which corresponds to a non-cluster
        if clusterNums[0] == 0:
            # note that the check is necessary because it can happen that the
            # entire observe data matrix exceeds the threshold, in that case we
            # don't want to remove the first element (which will be 1 instead
            # of zero)
            clusterNums = clusterNums[1:]
            clusterStats = clusterStats[1:]

        # use cluster sum instead of size
        clusterStats = [np.sum(dat[labelled == x]) for x in clusterNums]

        if getinds:
            return clusterStats, clusterNums, labelled
        else:
            return clusterStats

    # get observed clusters and maximum randomized clusterstats
    clusObs, clusNums, labelled = compute_clusterstats(obsdat)
    clusRnd = np.asarray([compute_clusterstats(rnddat[..., x], False)
                          for x in range(rnddat.shape[-1])])
    # treat randomizations with 0 cluster candidates as if their max was 0
    mymax = lambda x: 0 if len(x) == 0 else np.max(x)
    clusRnd = [mymax(x) for x in clusRnd]
    clusRnd.sort()

    # compare obs to rnd to obtain p-values
    for stat, num in zip(clusObs, clusNums):
        if tail == 0:
            if stat < 0:
                p[labelled == num] = np.searchsorted(clusRnd, stat) / rnddat.shape[-1]
            elif stat > 0:
                p[labelled == num] = 1 - np.searchsorted(clusRnd, stat) / rnddat.shape[-1]
        elif tail == -1:
            p[labelled == num] = np.searchsorted(clusRnd, stat) / rnddat.shape[-1]
        elif tail == 1:
            p[labelled == num] = 1 - np.searchsorted(clusRnd, stat) / rnddat.shape[-1]

    return p, labelled


def compute_perm_stats(obs_parts, rnd_parts, tails, if_smooth=False):
    clu_times, clu_lables = [], []
    for _ in range(obs_parts.shape[0]):
        if if_smooth:
            obs = smooth(obs_parts[_, :])
            rnd = np.array([smooth(rnd_parts[_, rep, :]) for rep in range(rnd_parts.shape[1])])
        else:
            obs = obs_parts[_, :]
            rnd = rnd_parts[_, :, :]
        clu_t, clu_lable = permutation_test(obs, rnd.T, tail=tails[_])
        clu_times.append(clu_t)
        clu_lables.append(clu_lable)

    return clu_times, clu_lables


def melt_data(data, models, times):
    dat_models = []
    for a, m in enumerate(models):
        dat = pd.DataFrame(data[a, :].T)
        # melt data into a long format
        dat['time (s)'] = times
        dat['Model'] = m  # add the model label before melting
        dat_models.append(pd.melt(dat, id_vars=['time (s)', 'Model']))

    # combine every model into long format
    data_melt = pd.concat(dat_models)

    return data_melt


def get_slices(clu_t, clu_lable, times):
    clusters = np.unique(clu_lable)
    clusters = clusters[clusters > 0]
    result = np.zeros((len(clusters), 3))

    for _ in range(len(clusters)):
        idc = clu_lable == _ + 1
        p_value = np.unique(clu_t[idc])[0]
        slices = times[idc]
        result[_, :] = p_value, slices[0], slices[-1]

    return result


def plot_perm_results(obs, clt_times, clt_labels, model_names, part_names, times=np.linspace(-0.5, 2, 250),
                      threshold=0.050001):
    data_all_parts = []
    for m in range(len(model_names)):
        data_melted = melt_data(obs[m], models=part_names, times=times)
        data_melted['variable'] = model_names[m]
        data_all_parts.append(data_melted)
    df = pd.concat(data_all_parts)
    df['decoding accuracy'] = df['value']
    sns.set_style("ticks")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    cols = sns.cubehelix_palette(len(clt_times[0]), rot=-.25, light=.7)
    g = sns.FacetGrid(df, col="variable", hue="Model", palette=cols, legend_out=True)
    g.map(sns.lineplot, "time (s)", "decoding accuracy")

    for m in range(len(model_names)):
        g.axes[0][m].axhline(0, linestyle='--', linewidth=0.8, color='black')
        g.axes[0][m].axhline(0.5, linestyle='--', linewidth=0.8, color='black')
        g.axes[0][m].axvline(0, linestyle='--', linewidth=0.8, color='black')
        g.axes[0][m].axvline(0.5, linestyle='--', linewidth=0.8, color='black')
        g.axes[0][m].axvline(1., linestyle='--', linewidth=0.8, color='black')

        for p in range(len(clt_times[0])):
            slices = get_slices(clt_times[m][p], clt_labels[m][p], times)
            for s_i in range(slices.shape[0]):
                if slices[s_i, 0] <= threshold:
                    g.axes[0][m].hlines(xmin=slices[s_i, 1], xmax=slices[s_i, 2], colors=cols[p],
                                        y=0.45 - (p / 30),
                                        linewidth=4)
    # g.set_titles(row_template='', col_template='')
    # plt.ylim([-10, 10]
    g.set_titles("{col_name}")
    # g.axes[0][-1].legend(loc='lower left')

    return


def plot_clusters(ax, clt_times, clt_labels, times, epoch_names, colour_lis, p_threshold=0.05, plot_chance_lvl=0.45):
    for p in range(len(epoch_names)):
        slices = get_slices(clt_times[p], clt_labels[p], times)
        for s_i in range(slices.shape[0]):
            if slices[s_i, 0] <= p_threshold:
                if p == 2:
                    ax.hlines(xmin=slices[s_i, 1], xmax=slices[s_i, 2], colors=colour_lis[p],
                              y=plot_chance_lvl - (p / 80), linestyles=(0, (1, 0.5)),
                              linewidth=2)
                else:
                    ax.hlines(xmin=slices[s_i, 1], xmax=slices[s_i, 2], colors=colour_lis[p],
                              y=plot_chance_lvl - (p / 80),
                              linewidth=2)

    return


def p_into_stars(p_val):
    if p_val <= 0.001:
        stars = "***"
    elif p_val <= 0.01:
        stars = "**"
    elif p_val <= 0.05:
        stars = "*"
    elif (p_val > 0.05) & (p_val <= 0.1):
        stars = '†'
    elif p_val > 0.1:
        stars = 'ns'

    return stars


def reg_par_recovery(n_neurons=100, max_noise=5, n_trials=101, fit_to_noise=False, n_bootstraps=100):
    noise_levels = np.linspace(0.0001, max_noise, 9)
    corrs_min = np.zeros((2, len(noise_levels), n_bootstraps))
    r2_min = np.zeros((2, len(noise_levels), n_bootstraps))
    corrs_rnd = np.zeros((2, len(noise_levels), n_bootstraps))
    r2_rnd = np.zeros((2, len(noise_levels), n_bootstraps))

    coefs_min_lis = []
    coefs_rnd_lis = []
    print('Recovering the underlying covariance matrix')
    for sig in tqdm(range(len(noise_levels))):
        for n_bootstrap in range(n_bootstraps):

            # Structured selectivity
            cov = np.zeros((3, 3))
            cov[2, 2] = 1

            # simulate neuronal selectivity profiles of minimal
            s_opt = np.random.multivariate_normal(np.zeros(3), cov, n_neurons)

            # simulate neuronal selectivity profiles of random
            s_rnd = np.random.multivariate_normal(np.zeros(3), np.diag([1 / 3, 1 / 3, 1 / 3]), n_neurons)

            design = np.zeros((4, 3))
            design[0, 0] = -0.5
            design[0, 1] = 0.5
            design[0, 2] = -0.5
            design[1, 0] = 0.5
            design[1, 1] = -0.5
            design[1, 2] = -0.5
            design[2, :] = -0.5
            design[2, 2] = 0.5
            design[3, :] = 0.5  # rewarded

            # construct design matrix populated with orthogonalised coefficients
            design_stack = np.tile(design.T, n_trials).T

            # prepare the regression models
            clf1 = linear_model.LinearRegression(fit_intercept=True)
            clf2 = linear_model.LinearRegression(fit_intercept=True)

            # generate firing rate for the optimal model using orthogonalised coefficients
            r_min = s_opt @ design_stack.T
            r_rnd = s_rnd @ design_stack.T

            # add noise to the firing rates
            r_train_min = r_min.T + np.random.normal(0, noise_levels[sig], (4 * n_trials, n_neurons))
            r_train_rnd = r_rnd.T + np.random.normal(0, noise_levels[sig], (4 * n_trials, n_neurons))

            # try to recover the model when only noise was supplied
            if fit_to_noise:
                r_train_min = np.random.normal(0, noise_levels[sig], (4 * n_trials, n_neurons))
                r_train_rnd = np.random.normal(0, noise_levels[sig], (4 * n_trials, n_neurons))

            design = np.zeros((4, 3))
            design[0, 0] = -0.5;
            design[0, 1] = 0.5;
            design[0, 2] = -0.5
            design[1, 0] = 0.5;
            design[1, 1] = -0.5;
            design[1, 2] = -0.5
            design[2, :] = -0.5;
            design[2, 2] = 0.5
            design[3, :] = 0.5  # rewarded
            design_stack = np.tile(design.T, n_trials).T

            # fit regression to data (underlying structured selectivity) using orthogonalised coefficients
            clf1.fit(design_stack, r_train_min)
            clf2.fit(design_stack, r_train_rnd)

            # get and mean-center orthogonalised coefficients
            coefs_min = clf1.coef_
            coefs_min = coefs_min - np.mean(coefs_min, axis=0, keepdims=True)
            coefs_rnd = clf2.coef_
            coefs_rnd = coefs_rnd - np.mean(coefs_rnd, axis=0, keepdims=True)

            # get true covariance under min generative data
            true_min_min = np.zeros((3, 3))
            true_min_min[2, 2] = np.cov(coefs_min.T)[2, 2]

            true_min_min = true_min_min.flatten()
            m_rnd = np.mean(np.diag(np.cov(coefs_min.T)))
            true_min_rnd = np.diag([m_rnd, m_rnd, m_rnd]).flatten()

            r2_min[0, sig, n_bootstrap] = r2_score(true_min_min, np.cov(coefs_min.T).flatten())
            corrs_min[0, sig, n_bootstrap] = pearsonr(true_min_min, np.cov(coefs_min.T).flatten())[0]

            r2_min[1, sig, n_bootstrap] = r2_score(true_min_rnd, np.cov(coefs_min.T).flatten())
            corrs_min[1, sig, n_bootstrap] = pearsonr(true_min_rnd, np.cov(coefs_min.T).flatten())[0]

            # get true covariance under rnd generative data
            true_rnd_min = np.zeros((3, 3))
            true_rnd_min[2, 2] = np.cov(coefs_rnd.T)[2, 2]
            true_rnd_min = true_rnd_min.flatten()

            m_rnd = np.mean(np.diag(np.cov(coefs_rnd.T)))
            true_rnd_rnd = np.diag([m_rnd, m_rnd, m_rnd]).flatten()

            r2_rnd[0, sig, n_bootstrap] = r2_score(true_rnd_min, np.cov(coefs_rnd.T).flatten())
            corrs_rnd[0, sig, n_bootstrap] = pearsonr(true_rnd_min, np.cov(coefs_rnd.T).flatten())[0]

            r2_rnd[1, sig, n_bootstrap] = r2_score(true_rnd_rnd, np.cov(coefs_rnd.T).flatten())
            corrs_rnd[1, sig, n_bootstrap] = pearsonr(true_rnd_rnd, np.cov(coefs_rnd.T).flatten())[0]

        coefs_min_lis.append(coefs_min)
        coefs_rnd_lis.append(coefs_rnd)
    return corrs_min, r2_min, corrs_rnd, r2_rnd, coefs_min_lis, coefs_rnd_lis


def grab_variables(Name):
    with open(Name + '.txt', "rb") as f:
        data = pickle.load(f)
    total_cost_over_time = data[1]
    betas_final = data[0][0]
    w_final = data[0][1]
    w_b_final = data[0][2]
    perf_cost_final = data[0][3]
    reg_cost_final = data[0][4]
    del data
    return betas_final, w_final, w_b_final, perf_cost_final, reg_cost_final, total_cost_over_time


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def epairs_metric(weights1, weights2, l=20):
    def epairs(weights, l=5):
        N = weights.shape[0]
        angles = np.zeros((N))
        for n in range(N):
            cosine = weights[n, :] @ weights.T / (
                    np.linalg.norm(weights[n, :]) *
                    np.linalg.norm(weights, axis=1))
            # cosine = np.abs(cosine)
            cosine = np.delete(cosine, n)
            cosine.sort()
            angles[n] = np.median(np.arccos(cosine[-l:]))
        return angles

    return np.abs(np.mean(epairs(weights1, l=l)) - np.mean(epairs(weights2, l=l)))


def split_data(data, labels1, n_splits=10, min_trl=100, n_condi=8):
    data1_splits = np.zeros((n_splits, int((min_trl / n_splits) * n_condi), data.shape[1], 250))

    for c in range(n_condi):
        data1 = data[labels1 == c, :, :]
        trl_idc = np.repeat(list(range(n_splits)), int(min_trl / n_splits))
        random.shuffle(trl_idc)

        for split in range(n_splits):
            data1_splits[split, c * 10:c * 10 + 10, :, :] = data1[trl_idc == split, :, :]

    return data1_splits


def get_sd_dimensions(data, labels, time_window, method='svm', n_jobs=None, n_inter=1, n_condi=16):
    all_combos = list(itertools.combinations(list(range(n_condi)), int(n_condi / 2)))
    n_combos = int(len(all_combos) / 2)
    decoding_targets = np.zeros((n_combos, n_condi))
    for i in range(n_combos):
        decoding_targets[i, all_combos[i]] = 1

    n_combos = decoding_targets.shape[0]
    scores = []
    print('Cutting along all possible axes (shattering dimensionality)')
    for combo in tqdm(range(n_combos)):
        if n_jobs != None:
            X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=True)
        else:
            X = np.mean(data[:, :, time_window[0]:time_window[1]], axis=-1, keepdims=False)
        y = assign_lables(labels, decoding_targets[combo, :])
        scores.append(decode(X, y, method=method, n_jobs=n_jobs, n_inter=n_inter))

    return np.array(scores)


def get_sd_min_axis(data, mode='momentum'):
    from scipy.optimize import curve_fit

    def func_sigmoid(x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    axes_min = np.zeros(data.shape[0])
    for rep in range(data.shape[0]):

        if mode == 'momentum':
            data_sorted = np.array(sorted(data[rep, :], reverse=True))
            axes_min[rep] = next((i for i, j in enumerate(data_sorted < 0) if j), None)
        elif mode == 'sigmoid':
            data_sorted = np.array(sorted(data[rep, :], reverse=False))
            x = data_sorted
            y = np.array(list(range(1, data_sorted.shape[0] + 1)))
            p0 = [max(y), np.median(x), 1, min(y)]
            axes_min[rep] = curve_fit(func_sigmoid, x, y, p0, maxfev=5000)[0][1]

    return axes_min


def get_xgen_exp2(data, labels, condi_fac, target_var_fac, method='svm'):
    condi_lis1 = condi_fac[target_var_fac == 0]
    condi_lis2 = condi_fac[target_var_fac == 1]

    combos = []
    for x in condi_lis1:
        for y in condi_lis2:
            combos.append([x, y])
    train_test_combos_unfltr = list(itertools.combinations(combos, 2))

    # filter for duplicates
    train_test_combos = []
    for _ in range(len(train_test_combos_unfltr)):
        lis = list(np.array(train_test_combos_unfltr[_]).flatten())
        has_dubs = len(lis) != len(np.unique(lis))
        if not has_dubs:
            train_test_combos.append(train_test_combos_unfltr[_])

    n_combos = len(train_test_combos)
    scores = []
    print('Cutting along all possible axes (xgen)')
    for combo in tqdm(range(n_combos)):
        train_condi = train_test_combos[combo][0]
        test_condi = train_test_combos[combo][1]
        idc_train = np.in1d(labels, train_condi)
        idc_test = np.in1d(labels, test_condi)
        X1 = data[idc_train, :]
        X2 = data[idc_test, :]
        y = np.concatenate([np.zeros(int(X1.shape[0] / 2)), np.ones(int(X1.shape[0] / 2))])
        scores.append(decode_xgen(X1, X2, y, y, method=method, n_jobs=None))

    return np.mean(scores)


def cross_val_pca(data, labels, factor, n_comps, n_splits=10):
    # zscore the data
    data = (data - np.mean(data, axis=0, keepdims=True)) / (np.std(data, axis=0, keepdims=True) + 1)
    labels = np.array(assign_lables(labels, factor))

    n_trls = data.shape[0]
    idx_rnd = np.concatenate([np.zeros(n_trls // 2), np.ones(n_trls // 2)])
    if (n_trls % 2) > 0:
        idx_rnd = np.concatenate([idx_rnd, [1.0]])

    var_ratios_splits = []
    for split in range(n_splits):
        random.shuffle(idx_rnd)
        data1 = data[idx_rnd == 0, :, :]
        data2 = data[idx_rnd == 1, :, :]

        labels1 = labels[idx_rnd == 0]
        labels2 = labels[idx_rnd == 1]

        data1 = condi_avg(data1, labels1)
        data2 = condi_avg(data2, labels2)

        dat1 = data1.mean(-1)
        dat2 = data2.mean(-1)
        pca1 = PCA(n_components=n_comps, random_state=42)
        pca2 = PCA(n_components=n_comps, random_state=42)

        pca1.fit(dat1)
        comps = pca1.transform(dat2)
        var_ratio1 = pca1.explained_variance_ratio_

        pca2.fit(dat2)
        comps = pca2.transform(dat1)
        var_ratio2 = pca2.explained_variance_ratio_

        var_ratios_splits.append(np.array([var_ratio1, var_ratio2]).mean(0))

    var_ratio = np.array(var_ratios_splits).mean(0)

    return var_ratio


def set_seed(seed=None):
    """
    Function that controls randomness. NumPy and random modules must be imported.

    Args:
      seed : Integer
        A non-negative integer that defines the random state. Default is `None`.

    Returns:
      Nothing.
    """
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

    print(f'Random seed {seed} has been set.')


def equalise_data_witihn_session(data, labels, which_trl='end', n_splits=4):
    trl_min_ses = np.min([data[i].shape[0] for i in range(len(data))])

    if which_trl == 'beginning':
        data_re = [data[_][:trl_min_ses, :, :] for _ in range(len(data))]
        labels_re = [labels[_][:trl_min_ses] for _ in range(len(data))]
    elif which_trl == 'end':
        data_re = [data[_][-trl_min_ses:, :, :] for _ in range(len(data))]
        labels_re = [labels[_][-trl_min_ses:] for _ in range(len(data))]
    elif which_trl == 'rnd':
        idx_re = np.random.choice(np.array(list(range(0, data.shape[0]))), trl_min_ses, replace=False)
        data_re = [data[_][idx_re, :, :] for _ in range(len(data))]
        labels_re = [labels[_][idx_re] for _ in range(len(data))]
    elif which_trl == 'middle':
        labels_re = []
        data_re = []
        for i_sess in range(len(data)):
            idc_half = int(len(labels[i_sess]) / 2)
            n_trls = len(labels[i_sess])
            if (n_trls % 2) > 0:
                labels_re.append(labels[i_sess][idc_half - int(n_trls / 2):idc_half + int(n_trls / 2) + 1])
                data_re.append(data[i_sess][idc_half - int(n_trls / 2):idc_half + int(n_trls / 2) + 1, :, :])
            else:
                labels_re.append(labels[i_sess][idc_half - int(n_trls / 2):idc_half + int(n_trls / 2)])
                data_re.append(data[i_sess][idc_half - int(n_trls / 2):idc_half + int(n_trls / 2), :, :])

    min_condi = int(
        np.min([np.min(list(get_freqs(labels_re[i]).values())) for i in range(len(labels_re))]) / n_splits) - 1

    n_trl_bck = int(trl_min_ses / n_splits)
    trial_start = 0
    trial_stop = n_trl_bck
    labels_all_blcks = []
    data_all_blcks = []
    for split in range(n_splits):
        data_block = [data_re[i][trial_start:trial_stop, :, :] for i in range(len(data_re))]
        label_block = [labels_re[i][trial_start:trial_stop] for i in range(len(data_re))]

        data_blck, label_blck = prepare_data(data_block, label_block, which_trl='beginning', set_min_trl=min_condi)
        trial_start += n_trl_bck
        trial_stop += n_trl_bck

        data_all_blcks.append(data_blck)
        labels_all_blcks.append(label_blck)

    min_trl_blck_post = np.min([data_all_blcks[i].shape[0] for i in range(n_splits)])
    data_eq = np.array([data_all_blcks[i][:min_trl_blck_post, :, :] for i in range(n_splits)])
    labels_eq = np.array([labels_all_blcks[i][:min_trl_blck_post] for i in range(n_splits)])

    return data_eq, labels_eq


def compute_95ci(data, n_bootstraps=10000):
    # Bootstrap resampling
    bootstrap_means = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)

    # Compute the 95% confidence interval
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)

    return lower_bound, upper_bound


def get_xgen_cross_set_null(X1, X2, labels1, labels2, n_reps=100, method='svm', n_jobs=None):
    xgen_null = np.zeros(n_reps)
    print('Computing null distribution (cross-variable generalisation)')
    for _ in tqdm(range(n_reps)):
        labels2_rnd = labels2.copy()
        np.random.shuffle(labels2_rnd)
        xgen_null[_] = decode_xgen_within_ses(X1, X2, labels1, labels2_rnd, method=method, n_jobs=n_jobs)
    return xgen_null


def compute_p_value(obs1, obs2, rnd1, rnd2, tail='greater'):
    diff = obs2 - obs1
    diff_rnd = rnd2 - rnd1
    if tail == 'smaller':
        p_val = np.sum(diff_rnd > diff) / diff_rnd.shape[0]
    elif tail == 'greater':
        p_val = np.sum(diff_rnd < diff) / diff_rnd.shape[0]
    elif tail == 'two':
        p_val = 2 * np.sum(diff_rnd > np.abs(diff)) / diff_rnd.shape[0]
    else:
        raise ValueError('tail must be greater, smaller or two')

    print('     Stats: M1 = ', str(round(obs1, 3)), ', M2 = ', str(round(obs2, 3)), ' | p-value = ', str(round(p_val, 3)))

    return p_val


def run_within_session_decoding(X, X_times, labels, variables, N_SPLITS, N_REPS, TIME_WINDOW, splitting_factor):
    n_variables = len(variables)

    splitting_labels = np.array(assign_lables(labels[0, :], factor=splitting_factor))

    X_set1 = X[:, splitting_labels == 0, :]
    labels_set1 = labels[:, splitting_labels == 0]

    X_set2 = X[:, splitting_labels == 1, :]
    labels_set2 = labels[:, splitting_labels == 1]

    decoding = np.zeros((n_variables, 2, N_SPLITS))
    decoding_nulls = np.zeros((n_variables, 2, N_SPLITS, N_REPS))

    for i_var in range(n_variables):
        for i_block in range(N_SPLITS):
            decoding[i_var, 0, i_block] = decode(X[i_block, :, :],
                                                 assign_lables(labels[i_block], factor=variables[i_var]), method='svm',
                                                 n_inter=40)
            decoding_nulls[i_var, 0, i_block, :] = get_decoding_null(X_times[i_block, :, :, :],
                                                                     assign_lables(labels[i_block],
                                                                                   factor=variables[i_var]),
                                                                     TIME_WINDOW,
                                                                     n_jobs=None, method='svm', n_reps=N_REPS)
            decoding[i_var, 1, i_block] = decode_xgen_within_ses(X_set1[i_block, :, :], X_set2[i_block, :, :],
                                                                 assign_lables(labels_set1[i_block],
                                                                               factor=variables[i_var][
                                                                                   splitting_factor == 0]),
                                                                 assign_lables(labels_set2[i_block],
                                                                               factor=variables[i_var][
                                                                                   splitting_factor == 0]),
                                                                 method='svm', n_jobs=None)

            decoding_nulls[i_var, 1, i_block, :] = get_xgen_cross_set_null(X_set1[i_block, :, :], X_set2[i_block, :, :],
                                                                           assign_lables(labels_set1[i_block],
                                                                                         factor=variables[i_var][
                                                                                             splitting_factor == 0]),
                                                                           assign_lables(labels_set2[i_block],
                                                                                         factor=variables[i_var][
                                                                                             splitting_factor == 0]),
                                                                           method='svm', n_jobs=None, n_reps=N_REPS)

    return decoding, decoding_nulls


def plot_within_session_decoding(decoding, decoding_nulls, variable_names, learning_stages, trails, ylim=[0.4, 0.8],
                                 plot_p=True):
    fig, ax = plt.subplots(1, len(variable_names), figsize=(2.5 * len(variable_names), 2.5))
    for i_var, variables in enumerate(variable_names):
        ax[i_var].plot(learning_stages, decoding[i_var, 0, :], label='decoding', color='black')
        ax[i_var].plot(learning_stages, decoding[i_var, 1, :], label='cross-gen.\ndecoding', color='grey', zorder=-5)
        ax[i_var].scatter(learning_stages, decoding[i_var, 0, :], color='black')
        ax[i_var].scatter(learning_stages, decoding[i_var, 1, :], color='grey', zorder=-5)
        ax[i_var].set_title(variables)
        ax[i_var].set_ylim(ylim)
        ax[i_var].set_xlabel('learning stage')
        ax[i_var].set_ylabel('accuracy')
        ax[i_var].axhline(0.5, color='black', linestyle='--')
        sns.despine(top=True, right=True)
        # plot p-values for decoding and cross-gen decoding
        if plot_p:
            p_val_dec = compute_p_value(decoding[i_var, 0, -1], decoding[i_var, 0, 0], decoding_nulls[i_var, 0, -1],
                                        decoding_nulls[i_var, 0, 1], tail=trails[i_var])
            p_val_cross = compute_p_value(decoding[i_var, 1, -1], decoding[i_var, 1, 0], decoding_nulls[i_var, 1, -1],
                                          decoding_nulls[i_var, 1, 1], tail=trails[i_var])
            ax[i_var].text(0.5, 0.75, 'p = ' + str(np.round(p_val_dec, 3)), fontsize=8, transform=ax[i_var].transAxes,
                           ha='center', color='black')
            ax[i_var].text(0.5, 0.65, 'p = ' + str(np.round(p_val_cross, 3)), fontsize=8, transform=ax[i_var].transAxes,
                           ha='center', color='grey')
    ax[0].legend()
    plt.tight_layout()
    plt.show()


def decode_xgen_within_ses(X1, X2, y1, y2, method='svm', n_jobs=-1, n_iter=10):
    # prepare a series of classifier applied at each time sample
    if method == 'svm':
        clf1 = make_pipeline(StandardScaler(), SVM(C=5e-4))
        if n_jobs != None:
            clf1 = SlidingEstimator(clf1, verbose=False, n_jobs=-1)
        clf2 = make_pipeline(StandardScaler(), SVM(C=5e-4))
        if n_jobs != None:
            clf2 = SlidingEstimator(clf2, verbose=False, n_jobs=-1)
    elif method == 'lda':
        clf1 = make_pipeline(StandardScaler(), LDA())
        if n_jobs != None:
            clf1 = SlidingEstimator(clf1, verbose=False, n_jobs=-1)
        clf2 = make_pipeline(StandardScaler(), LDA())
        if n_jobs != None:
            clf2 = SlidingEstimator(clf2, verbose=False, n_jobs=-1)

    scores = np.zeros((n_iter, 2))
    for _ in range(n_iter):
        idc1 = np.arange(X1.shape[0])
        idc2 = np.arange(X2.shape[0])
        idx_rnd1 = np.random.choice(idc1, X1.shape[0], replace=True)
        idx_rnd2 = np.random.choice(idc2, X2.shape[0], replace=True)
        X1_rnd = X1[idx_rnd1, :]
        y1_rnd = np.array(y1)[idx_rnd1]
        X2_rnd = X2[idx_rnd2, :]
        y2_rnd = np.array(y2)[idx_rnd2]

        clf1.fit(X1_rnd, y1_rnd)
        scores[_, 0] = clf1.score(X2_rnd, y2_rnd)

        clf2.fit(X2_rnd, y2_rnd)
        scores[_, 1] = clf2.score(X1_rnd, y1_rnd)

    return scores.mean((1, 0))


def split_into_blocks(dat, labels, n_splits):
    n_trl_blck = int(dat.shape[0] / n_splits)
    i_start = 0
    i_stop = n_trl_blck
    data_blocks = []
    labels_blocks = []
    for i_block in range(n_splits):
        data_blocks.append(dat[i_start:i_stop, :, :])
        labels_blocks.append(labels[i_start:i_stop])
        i_start += n_trl_blck
        i_stop += n_trl_blck
    return data_blocks, labels_blocks


def split_vector(input_vector, k, d):
    n = len(input_vector)  # Total number of elements in the input vector
    output_vectors = []

    # Calculate the step between the starts of each output vector to distribute elements evenly
    step = max(1, (n - d) // (k - 1)) if k > 1 else 0

    # Generate the output vectors
    for i in range(k):
        start_index = i * step
        end_index = start_index + d
        # Adjust the end index if it goes beyond the input vector length
        if end_index > n:
            start_index = max(0, n - d)  # Move back to fit the last vector
            end_index = n
        output_vector = input_vector[start_index:end_index]
        output_vectors.append(output_vector)
        if end_index == n:  # Stop if the last vector reaches the end of the input vector
            break

    return output_vectors


def split_data_blocks_moveavg(data, labels, N_SPLITS, N_WINDOWS):
    dat_split, labels_split = [], []
    for i_sess in range(len(data)):
        dat_split_ses, labels_split_ses = split_into_blocks(data[i_sess], labels[i_sess], N_SPLITS)
        dat_split.append(dat_split_ses)
        labels_split.append(labels_split_ses)

    dat_blocks, labs_blocks = [], []
    for i_block in range(N_SPLITS):
        dat_blocks.append([dat_split[_][i_block] for _ in range(len(data))])
        labs_blocks.append([labels_split[_][i_block] for _ in range(len(data))])

    data_blocks, labels_blocks = [], []
    for i_block in range(N_SPLITS):
        dat = dat_blocks[i_block]
        labs = labs_blocks[i_block]
        trl_number_min = np.min([dat[_].shape[0] for _ in range(len(dat))])
        dat_block_windows = []
        labels_block_windows = []
        for i_sess in range(len(dat)):
            dat_ses = dat[i_sess]
            labs_ses = labs[i_sess]
            idc_trials = np.arange(dat_ses.shape[0])
            idc_windows = split_vector(idc_trials, N_WINDOWS, trl_number_min - N_WINDOWS)
            dat_block_windows.append([dat_ses[idc_windows[_], :, :] for _ in range(N_WINDOWS)])
            labels_block_windows.append([labs_ses[idc_windows[_]] for _ in range(N_WINDOWS)])

        data_pseudopop_wins, labs_pseudopop_wins = [], []
        for i_window in range(N_WINDOWS):
            dat_window = [dat_block_windows[_][i_window] for _ in range(len(dat))]
            labels_window = [labels_block_windows[_][i_window] for _ in range(len(dat))]

            data_pseudopop, labs_pseudopop = prepare_data(dat_window, labels_window, which_trl="end")
            data_pseudopop_wins.append(data_pseudopop)
            labs_pseudopop_wins.append(labs_pseudopop)

        data_pseudopop_wins = np.array(data_pseudopop_wins)
        labs_pseudopop_wins = np.array(labs_pseudopop_wins)

        data_blocks.append(data_pseudopop_wins)
        labels_blocks.append(labs_pseudopop_wins)

    return data_blocks, labels_blocks


def split_data_blocks_moveavg_sel(data, labels, N_SPLITS):
    dat_split, labels_split = [], []
    for i_sess in range(len(data)):
        dat_split_ses, labels_split_ses = split_into_blocks(data[i_sess], labels[i_sess], N_SPLITS)
        dat_split.append(dat_split_ses)
        labels_split.append(labels_split_ses)

    dat_blocks, labs_blocks = [], []
    for i_block in range(N_SPLITS):
        dat_blocks.append([dat_split[_][i_block] for _ in range(len(data))])
        labs_blocks.append([labels_split[_][i_block] for _ in range(len(data))])

    return dat_blocks, labs_blocks


def compute_p_val_learning(data_first, data_last, n_perm=10000, tail='greater'):
    mean_diff = np.mean(data_last) - np.mean(data_first)
    data_all = np.concatenate((data_first, data_last))
    mean_diff_rnd = []
    for _ in range(n_perm):
        idc_rnd = np.random.permutation(len(data_all))
        data_first_rnd = data_all[idc_rnd[:len(data_first)]]
        data_last_rnd = data_all[idc_rnd[len(data_first):]]
        mean_diff_rnd.append(np.mean(data_last_rnd) - np.mean(data_first_rnd))
    if tail == 'greater':
        p_value = np.sum(mean_diff_rnd > mean_diff) / len(mean_diff_rnd)
    elif tail == 'less':
        p_value = np.sum(mean_diff_rnd < mean_diff) / len(mean_diff_rnd)
    elif tail == 'two-sided':
        p_value = np.sum(np.abs(mean_diff_rnd) > np.abs(mean_diff)) / len(mean_diff_rnd)
    return p_value


def plot_fixation_breaks(reward_prop, animal):
    plt.figure()
    sessions = list(range(1, len(reward_prop) + 1))
    plt.plot(sessions, reward_prop, label='No reward - reward')
    plt.legend()
    plt.xlabel('Session')
    plt.ylabel('Proportion of fixation breaks')
    plt.title(animal)
    plt.show()


def sum_values(trial_types, keys):
    return sum([trial_types.get(key, 0) for key in keys])


def compute_p_val_learning(data_first, data_last, n_perm=10000, tail='greater'):
    mean_diff = np.mean(data_last) - np.mean(data_first)
    data_all = np.concatenate((data_first, data_last))
    mean_diff_rnd = []
    for _ in range(n_perm):
        idc_rnd = np.random.permutation(len(data_all))
        data_first_rnd = data_all[idc_rnd[:len(data_first)]]
        data_last_rnd = data_all[idc_rnd[len(data_first):]]
        mean_diff_rnd.append(np.mean(data_last_rnd) - np.mean(data_first_rnd))
    if tail == 'greater':
        p_value = np.sum(mean_diff_rnd > mean_diff) / len(mean_diff_rnd)
    elif tail == 'less':
        p_value = np.sum(mean_diff_rnd < mean_diff) / len(mean_diff_rnd)
    elif tail == 'two-sided':
        p_value = np.sum(np.abs(mean_diff_rnd) > np.abs(mean_diff)) / len(mean_diff_rnd)
    return p_value



def plot_mean_and_ci(reward, no_reward, n_perm=10000, tail='greater'):
    plt.figure(figsize=(4, 3))
    x = list(range(1, len(reward) + 1))
    y1 = np.array([np.mean(reward[i], axis=0) for i in range(len(reward))])
    y2 = np.array([np.mean(no_reward[i], axis=0) for i in range(len(no_reward))])
    # plote 95% confidence interval
    y1_95ci = np.array([1.96 * np.std(reward[i], axis=0) / np.sqrt(len(reward[i])) for i in range(len(reward))])
    y2_95ci = np.array(
        [1.96 * np.std(no_reward[i], axis=0) / np.sqrt(len(no_reward[i])) for i in range(len(no_reward))])

    plt.plot(x, y1, label='reward', color='black')
    plt.errorbar(x, y1, yerr=y1_95ci, fmt='o', color='black')
    # plt.fill_between(x, y1 - y1_sem, y1 + y1_sem, alpha=0.5)
    plt.plot(x, y2, label='no reward', color='grey')
    plt.errorbar(x, y2, yerr=y2_95ci, fmt='o', color='grey')
    # plt.fill_between(x, y2 - y2_sem, y2 + y2_sem, alpha=0.5)

    p_value_norew = compute_p_val_learning(no_reward[0], no_reward[-1], n_perm=n_perm, tail=tail)
    p_value_rew = compute_p_val_learning(reward[0], reward[-1], n_perm=n_perm, tail=tail)

    # annotate the plot with p-value
    plt.text(0.1, 0.9, f'p-value no reward = {round(p_value_norew, 3)}\np-value reward = {round(p_value_rew, 3)}',
             horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.xlabel('learning stage')
    plt.xticks([1, 2, 3, 4])
    plt.ylabel('fixation breaks (%)')
    plt.legend(['reward', 'no reward'], loc='lower left')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()


def plot_mean_and_ci_prop(gs, fig, reward_prop, n_perm=10000, tail='greater', stat='reg', plot_comp_dat=False):
    ax = fig.add_subplot(gs)

    x = list(range(1, len(reward_prop) + 1))
    y1 = np.array([np.mean(reward_prop[i], axis=0) for i in range(len(reward_prop))])
    # plote 95% confidence interval
    y1_95ci = np.array(
        [1.96 * np.std(reward_prop[i], axis=0) / np.sqrt(len(reward_prop[i])) for i in range(len(reward_prop))])

    ax.plot(x, y1, color='black')
    ax.errorbar(x, y1, yerr=y1_95ci, fmt='o', color='black')
    if stat == 'reg':
        # flatten reward_prop and construct list with stage labels
        y_flat = list(itertools.chain.from_iterable(reward_prop))
        x_flat = [1] * len(reward_prop[0]) + [2] * len(reward_prop[1]) + [3] * len(reward_prop[2]) + [4] * len(
            reward_prop[3])
        slope, intercept, p_value, r_value = compute_lin_reg(x_flat, y_flat, n_perm=n_perm, tail=tail)

        # annotate the plot with p-value
        ax.text(0.1, 0.9, f'p-value = {round(p_value, 3)}\nr = {round(r_value, 3)}',
                horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    elif stat == 'ttest':
        p_value_rew_prop = compute_p_val_learning(reward_prop[0], reward_prop[-1], n_perm=n_perm, tail=tail)
        # annotate the plot with p-value
        ax.text(0.1, 0.9, f'p-value = {round(p_value_rew_prop, 3)}',
                horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    if plot_comp_dat != False:
        y2 = np.array([np.mean(plot_comp_dat[i], axis=0) for i in range(len(plot_comp_dat))])
        # plote 95% confidence interval
        y2_95ci = np.array(
            [1.96 * np.std(plot_comp_dat[i], axis=0) / np.sqrt(len(plot_comp_dat[i])) for i in
             range(len(plot_comp_dat))])

        ax.plot(x, y2, color='grey')
        ax.errorbar(x, y2, yerr=y2_95ci, fmt='o', color='grey')

    ax.set_xlabel('learning stage')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylabel('no reward/reward\nproporiton')
    sns.despine(top=True, right=True)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8)
    ax.set_title('trial termination')


def get_fixation_breaks(sessions_animals, experiment_label='exp1'):
    with open('config.yml', 'r') as f:
        configs = yaml.safe_load(f)

    code_labels = configs['TRIGGER_CODES']
    fix_breaks_animals = []
    for animal in range(len(sessions_animals)):
        sessions = sessions_animals[animal]
        fix_numers = np.zeros((len(sessions), 5))
        for s in range(len(sessions)):

            print('Session: ', s + 1)
            beh = io.loadmat(configs['PATHS']['in_template_beh'].format(sessions[s]))
            codes = beh['uecode']
            codes = list(codes[0, :])
            times = beh['timingms']
            times = list(times[0, :])

            size = len(codes)
            idx_list = [idx for idx, val in
                        enumerate(codes) if val == 6116]

            trials = [codes[i: j] for i, j in
                      zip([0] + idx_list, idx_list +
                          ([size] if idx_list[-1] != size else []))]

            trials = trials[1:]

            conditions = []
            for _ in range(len(trials)):

                if code_labels['CUE1_ON'] in trials[_] and code_labels['TARGET1_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(1)
                elif code_labels['CUE1_ON'] in trials[_] and code_labels['TARGET2_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(2)
                elif code_labels['CUE1_ON'] in trials[_] and code_labels['TARGET3_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(3)
                elif code_labels['CUE1_ON'] in trials[_] and code_labels['TARGET4_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(4)
                elif code_labels['CUE2_ON'] in trials[_] and code_labels['TARGET1_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(5)
                elif code_labels['CUE2_ON'] in trials[_] and code_labels['TARGET2_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(6)
                elif code_labels['CUE2_ON'] in trials[_] and code_labels['TARGET3_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(7)
                elif code_labels['CUE2_ON'] in trials[_] and code_labels['TARGET4_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(8)

                elif code_labels['CUE3_ON'] in trials[_] and code_labels['TARGET1_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(9)
                elif code_labels['CUE3_ON'] in trials[_] and code_labels['TARGET2_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(10)
                elif code_labels['CUE3_ON'] in trials[_] and code_labels['TARGET3_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(11)
                elif code_labels['CUE3_ON'] in trials[_] and code_labels['TARGET4_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(12)
                elif code_labels['CUE4_ON'] in trials[_] and code_labels['TARGET1_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(13)
                elif code_labels['CUE4_ON'] in trials[_] and code_labels['TARGET2_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(14)
                elif code_labels['CUE4_ON'] in trials[_] and code_labels['TARGET3_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(15)
                elif code_labels['CUE4_ON'] in trials[_] and code_labels['TARGET4_ON'] in trials[_] and (
                        code_labels['BREAK_TARGET_ERROR'] in trials[_]):
                    conditions.append(16)

                if code_labels['BREAK_CUE_ERROR'] in trials[_] or code_labels['BREAK_TARGET_ERROR'] in trials[_] or (
                        code_labels['BREAK_ERROR'] in trials[_]) or (code_labels['FIXATION_ERROR'] in trials[_]):
                    conditions.append(99)

                if code_labels['BREAK_CUE_ERROR'] in trials[_]:
                    conditions.append(999)

            all_trls = []
            for _ in range(len(trials)):
                if code_labels['CUE1_ON'] in trials[_]:
                    all_trls.append(1)
                elif code_labels['CUE2_ON'] in trials[_]:
                    all_trls.append(1)

                elif code_labels['CUE3_ON'] in trials[_]:
                    all_trls.append(2)
                elif code_labels['CUE4_ON'] in trials[_]:
                    all_trls.append(2)

            trial_types = {value: len(list(freq)) for value, freq in groupby(sorted(conditions))}
            trial_all = {value: len(list(freq)) for value, freq in groupby(sorted(all_trls))}

            if experiment_label == 'exp1':
                fix_numers[s, 0] = len(trials)
                fix_numers[s, 1] = sum_values(trial_types,
                                              [1, 2, 7, 8])  # number of fixation breaks in reward trials
                fix_numers[s, 2] = sum_values(trial_types,
                                              [3, 4, 5, 6])  # number of fixation breaks in no reward trials
                fix_numers[s, 3] = trial_types[99]  # number of trials with all fixation breaks
                fix_numers[s, 4] = trial_types[999]  # number of trials with cue fixation breaks
            elif experiment_label == 'exp2':
                fix_numers[s, 0] = len(trials)
                fix_numers[s, 1] = sum_values(trial_types,
                                              [9, 10, 15, 16])  # number of fixation breaks in reward trials
                fix_numers[s, 2] = sum_values(trial_types,
                                              [11, 12, 13, 14])  # number of fixation breaks in no reward trials
                fix_numers[s, 3] = trial_types[99]  # number of trials with all fixation breaks
                fix_numers[s, 4] = trial_types[999]  # number of trials with cue fixation breaks

        fix_breaks_animals.append(fix_numers)
    return fix_breaks_animals


def get_data_stages(observe_or_run='observe', file_name=None, return_data=False, session_list=None):
    with open('config.yml', 'r') as file:
        configs = yaml.safe_load(file)

    if observe_or_run == 'run':

        data_all_parts, labels = get_data(session_list=session_list,
                                          path_spikes=configs['PATHS']['out_template_spks'],
                                          path_meta=configs['PATHS']['out_template_meta'],
                                          window=[0, 250],
                                          cut_off=None
                                          )

        data, _, _ = exclude_neurons(data=data_all_parts,
                                     session_list=session_list,
                                     path_locations=configs['PATHS']['out_template_loc'],
                                     path_sel_exclude=configs['PATHS']['out_template_sel_list'],
                                     loc=configs['ANALYSIS_PARAMS']['SAMPLED_AREAS']
                                     )

        save_data([data, labels], ['data', 'labels'], configs['PATHS']['output_path'] + file_name + '.pickle')

    elif observe_or_run == 'observe':
        return_data = True
        obj_loaded = load_data(configs['PATHS']['output_path'] + file_name + '.pickle')
        data = obj_loaded['data']
        labels = obj_loaded['labels']

    if return_data:
        return data, labels


def combine_session_lists(mode='time', which_exp='exp1', combine_all=True):
    with open('config.yml', 'r') as file:
        configs = yaml.safe_load(file)
    if which_exp == 'exp1':
        animal1_ses_labels = configs['SESSION_NAMES']['sessions_womble_1']
        animal2_ses_labels = configs['SESSION_NAMES']['sessions_wilfred_1']
    elif which_exp == 'exp2':
        animal1_ses_labels = configs['SESSION_NAMES']['sessions_womble_2']
        animal2_ses_labels = configs['SESSION_NAMES']['sessions_wilfred_2']

    if mode == 'time':
        min_num_sessions = min(len(animal1_ses_labels), len(animal2_ses_labels))
        sesions_split_animal1 = np.array_split(animal1_ses_labels, min_num_sessions)
        sesions_split_animal2 = np.array_split(animal2_ses_labels, min_num_sessions)

        ses_com = []
        for _ in range(min_num_sessions):
            ses_com.append(list(sesions_split_animal1[_]))
            ses_com.append(list(sesions_split_animal2[_]))
        if combine_all:
            ses_com = list(chain.from_iterable(ses_com))
    elif mode == 'fix_bias':
        session_labels = [animal1_ses_labels, animal2_ses_labels]
        fix_breaks_animals = get_fixation_breaks(session_labels, experiment_label=which_exp)

        sessions_sorted = []
        for i_anim in range(2):
            no_reward_fix = fix_breaks_animals[i_anim][:, 2] / fix_breaks_animals[i_anim][:, 3]
            reward_fix = fix_breaks_animals[i_anim][:, 1] / fix_breaks_animals[i_anim][:, 3]
            dat = no_reward_fix / reward_fix

            # sort data_prop_s and get the indices of the sorting
            sort_idx = np.argsort(dat)
            sessions_sorted.append(np.array(session_labels[i_anim])[sort_idx])

        ses_wom = np.array_split(sessions_sorted[0], configs['ANALYSIS_PARAMS']['N_STAGES'])
        ses_wil = np.array_split(sessions_sorted[1], configs['ANALYSIS_PARAMS']['N_STAGES'])
        ses_com = []
        for _ in range(configs['ANALYSIS_PARAMS']['N_STAGES']):
            ses_com.append(list(ses_wom[_]) + list(ses_wil[_]))
        if combine_all:
            ses_com = list(chain.from_iterable(ses_com))

    elif mode == 'stages':

        ses_wom = np.array_split(animal1_ses_labels, configs['ANALYSIS_PARAMS']['N_STAGES'])
        ses_wil = np.array_split(animal2_ses_labels, configs['ANALYSIS_PARAMS']['N_STAGES'])
        ses_com = []
        for _ in range(configs['ANALYSIS_PARAMS']['N_STAGES']):
            ses_com.append(list(ses_wom[_]) + list(ses_wil[_]))
        if combine_all:
            ses_com = list(chain.from_iterable(ses_com))

    elif (mode == 'balanced') & (which_exp == 'exp1'):
        animal1_ses_labels = animal1_ses_labels[:8] + animal1_ses_labels[-8:]
        animal2_ses_labels = animal2_ses_labels[:4] + animal2_ses_labels[-4:]
        ses_wom = np.array_split(animal1_ses_labels, configs['ANALYSIS_PARAMS']['N_STAGES'])
        ses_wil = np.array_split(animal2_ses_labels, configs['ANALYSIS_PARAMS']['N_STAGES'])

        ses_com = []
        for _ in range(configs['ANALYSIS_PARAMS']['N_STAGES']):
            ses_com.append(list(ses_wom[_]))
            ses_com.append(list(ses_wil[_]))
        if combine_all:
            ses_com = list(chain.from_iterable(ses_com))

    elif (mode == 'balanced_stages') & (which_exp == 'exp1'):
        ses_wom = np.array_split(animal1_ses_labels, configs['ANALYSIS_PARAMS']['N_STAGES'])
        ses_wil = [animal2_ses_labels[:3], animal2_ses_labels[3:5], animal2_ses_labels[5:7], animal2_ses_labels[-3:]]

        ses_com = []
        for _ in range(configs['ANALYSIS_PARAMS']['N_STAGES']):
            ses_com.append(list(ses_wom[_]) + list(ses_wil[_]))
        if combine_all:
            ses_com = list(chain.from_iterable(ses_com))

    return ses_com


def plot_reg_prop(reward_prop, n_perm=10000, tail='greater'):
    plt.figure(figsize=(3, 2))

    # flatten reward_prop and construct list with stage labels
    y_flat = list(itertools.chain.from_iterable(reward_prop))
    x_flat = [1] * len(reward_prop[0]) + [2] * len(reward_prop[1]) + [3] * len(reward_prop[2]) + [4] * len(
        reward_prop[3])
    slope, intercept, p_value, r_value = compute_lin_reg(x_flat, y_flat, n_perm=n_perm, tail=tail)

    # plot regression
    sns.regplot(x=x_flat, y=y_flat, color='black', scatter_kws={'color': 'black'},
                line_kws={'color': 'black'})

    # annotate the plot with p-value
    plt.text(0.1, 0.9, f'p-value = {round(p_value, 3)}\nr = {round(r_value, 3)}',
             horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    plt.xlabel('learning stage')
    plt.xticks([1, 2, 3, 4])
    plt.ylabel('no reward/reward\nproporiton')
    sns.despine(top=True, right=True)
    plt.axhline(y=1, color='black', linestyle='--')
    plt.title('fixation breaks')
    plt.tight_layout()
    plt.show()


def compute_lin_reg(x, y, n_perm=1000, tail='greater'):
    A = np.vstack([x, np.ones(len(x))]).T
    results = np.linalg.lstsq(A, y, rcond=None)
    slope = results[0][0]
    intercept = results[0][1]

    null_slopes = []
    for _ in range(n_perm):
        x_perm = np.random.permutation(x)
        results = np.linalg.lstsq(np.vstack([x_perm, np.ones(len(x))]).T, y, rcond=None)
        null_slopes.append(results[0][0])

    if tail == 'greater':
        p_value = np.sum(null_slopes > slope) / len(null_slopes)
    elif tail == 'less':
        p_value = np.sum(null_slopes < slope) / len(null_slopes)
    elif tail == 'two-sided':
        p_value = np.sum(np.abs(null_slopes) > np.abs(slope)) / len(null_slopes)

    # compute the r value of the regression
    r_value = spearmanr(x, y)[0]

    return slope, intercept, p_value, r_value


def plot_regression(df, n_perm=100000):
    # Unique animals
    animals = df['animal'].unique()

    # Create subplots
    fig, axes = plt.subplots(1, len(animals), figsize=(5 * len(animals), 5), sharey=False)

    # Check if there is only one subplot (axis) and make it iterable
    if len(animals) == 1:
        axes = [axes]

    # Iterate over each animal and its corresponding axis
    for animal, ax in zip(animals, axes.flatten()):
        # Filter the DataFrame for the current animal
        animal_df = df[df['animal'] == animal]

        # Plot the regression for the current animal
        sns.regplot(x='x', y='y', data=animal_df, ax=ax, color='black', scatter_kws={'color': 'black'},
                    line_kws={'color': 'black'})

        # Compute the regression for the current animal
        slope, intercept, p_value, r_value = compute_lin_reg(animal_df['x'], animal_df['y'], n_perm=n_perm,
                                                             tail='greater')

        # Calculate buffer for x and y limits
        x_buffer = (animal_df['x'].max() - animal_df['x'].min()) * 0.1
        y_buffer = (animal_df['y'].max() - animal_df['y'].min()) * 0.1

        # Set individual x and y limits with buffer
        ax.set_xlim(animal_df['x'].min() - x_buffer, animal_df['x'].max() + x_buffer)
        ax.set_ylim(animal_df['y'].min() - y_buffer, animal_df['y'].max() + y_buffer)

        # Annotate the plot with p-value and r-value
        ax.text(0.5, 0.9, f'p-value = {round(p_value, 3)}\nr = {round(r_value, 3)}',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        # Set title
        ax.set_title(animal)
        ax.set_xlabel('Session')
        ax.set_ylabel('Proportion of fixation breaks')
        sns.despine(ax=ax, top=True, right=True)

    # Set common labels
    plt.tight_layout()
    plt.show()


def creat_plot_grid(n_rows, n_cols, size):
    fig = plt.figure(figsize=(size * n_cols, (size * n_rows) * 0.8))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    return fig, gs


def split_data_stages_moveavg(data, labels, n_stages, n_windows, trl_min=None, if_rnd=False):
    # collapse list of lists into a single list
    data_lis = list(chain.from_iterable(data))
    trl_number_min = np.min([data_lis[_].shape[0] for _ in range(len(data_lis))])

    data_stages, labels_stages = [], []
    for i_stage in range(n_stages):
        dat = data[i_stage]
        labs = labels[i_stage]
        dat_stage_windows = []
        labels_stage_windows = []
        for i_sess in range(len(dat)):
            dat_ses = dat[i_sess]
            labs_ses = labs[i_sess]
            idc_trials = np.arange(dat_ses.shape[0])
            idc_windows = split_vector(idc_trials, n_windows, trl_number_min - n_windows)
            dat_stage_windows.append([dat_ses[idc_windows[i_win], :, :] for i_win in range(n_windows)])
            labels_stage_windows.append([labs_ses[idc_windows[i_win]] for i_win in range(n_windows)])

        data_pseudopop_wins, labs_pseudopop_wins = [], []
        for i_window in range(n_windows):
            dat_window = [dat_stage_windows[_][i_window] for _ in range(len(dat))]
            labels_window = [labels_stage_windows[_][i_window] for _ in range(len(dat))]

            if trl_min is None:
                data_pseudopop, labs_pseudopop = prepare_data(dat_window, labels_window, which_trl="end")
            else:
                if if_rnd:
                    data_pseudopop, labs_pseudopop = prepare_data(dat_window, labels_window, which_trl="rnd",
                                                                  set_min_trl=trl_min)
                else:
                    data_pseudopop, labs_pseudopop = prepare_data(dat_window, labels_window, which_trl="end",
                                                                  set_min_trl=trl_min)
            data_pseudopop_wins.append(data_pseudopop)
            labs_pseudopop_wins.append(labs_pseudopop)

        data_pseudopop_wins = np.array(data_pseudopop_wins)
        labs_pseudopop_wins = np.array(labs_pseudopop_wins)

        data_stages.append(data_pseudopop_wins)
        labels_stages.append(labs_pseudopop_wins)

    return data_stages, labels_stages


def run_moving_window_decoding(data_eq, labels_eq, variables, time_window, method='svm', n_jobs=None, if_xgen=True,
                               if_null=False, n_reps=None):
    n_windows = data_eq[0].shape[0]
    n_stages = len(data_eq)
    shattering_dim = np.zeros((n_stages, 31, n_windows))
    decoding = np.zeros((n_stages, 4, n_windows))

    if if_xgen:
        xgen_decoding = np.zeros((n_stages, 4, n_windows))

    for i_stage in range(n_stages):
        for i_window in range(n_windows):
            X_stage = data_eq[i_stage][i_window, :, :, :]
            y_stage = labels_eq[i_stage][i_window, :]

            decoding_sess, decoding_dich_sess = get_decoding(data=X_stage,
                                                             labels=y_stage,
                                                             variables=variables,
                                                             time_window=time_window,
                                                             method=method,
                                                             n_jobs=n_jobs)
            if if_xgen:
                decoding_xgen_sess = get_xgen(data=X_stage,
                                              labels=y_stage,
                                              variables=variables,
                                              time_window=time_window,
                                              method=method,
                                              mode='only_rel',
                                              n_jobs=n_jobs,
                                              )
                xgen_decoding[i_stage, :, i_window] = decoding_xgen_sess

            shattering_dim[i_stage, :, i_window] = decoding_dich_sess
            decoding[i_stage, :, i_window] = decoding_sess[:4]

    shattering_dim = shattering_dim.mean(-1)
    decoding = decoding.mean(-1)
    if if_xgen:
        xgen_decoding = xgen_decoding.mean(-1)

    if if_null:
        decoding_null = np.zeros((n_stages, n_reps, n_windows))
        if if_xgen:
            xgen_null = np.zeros((n_stages, n_reps, 4, n_windows))
        for i_rep in tqdm(range(n_reps)):
            for i_stage in range(n_stages):
                X_stage = data_eq[i_stage]
                y_stage = labels_eq[i_stage]

                idc_rnd = np.arange(X_stage.shape[1])
                random.shuffle(idc_rnd)
                y_stage_rnd = y_stage[:, idc_rnd]

                for i_window in range(n_windows):
                    X_stage_win = X_stage[i_window, :, :, :]
                    y_stage_win = y_stage_rnd[i_window, :]

                    labels_random = assign_lables(y_stage_win, factor=[0, 0, 0, 0, 1, 1, 1, 1])

                    decoding_null[i_stage, i_rep, i_window] = decode(
                        X_stage_win[:, :, time_window[0]: time_window[1]].mean(-1), labels_random, method=method,
                        n_jobs=n_jobs)
                    if if_xgen:
                        xgen_null[i_stage, i_rep, :, i_window] = get_xgen(data=X_stage_win,
                                                                          labels=y_stage_win,
                                                                          variables=variables,
                                                                          time_window=time_window,
                                                                          method=method,
                                                                          mode='only_rel',
                                                                          n_jobs=n_jobs,
                                                                          verbose=False
                                                                          )

        decoding_null = decoding_null.mean(-1)
        if if_xgen:
            xgen_null = xgen_null.mean(-1)
    if if_xgen:
        if if_null:
            return shattering_dim, decoding, xgen_decoding, decoding_null, xgen_null
        else:
            return shattering_dim, decoding, xgen_decoding
    else:
        if if_null:
            return shattering_dim, decoding, decoding_null
        else:
            return shattering_dim, decoding


def shuffle_stages(stage1, stage4, label1):
    data_all = np.concatenate((stage1, stage4), axis=2)
    n_cells_stage1 = stage1.shape[2]
    n_cells_all = data_all.shape[2]

    idc_cells = np.arange(n_cells_all)
    random.shuffle(idc_cells)
    dat_stage1_shuffled = data_all[:, :, idc_cells[:n_cells_stage1], :]
    dat_stage4_shuffled = data_all[:, :, idc_cells[n_cells_stage1:], :]
    data_epochs1and4 = [dat_stage1_shuffled, dat_stage4_shuffled]
    labels_epochs1and4 = [label1, label1]

    return data_epochs1and4, labels_epochs1and4


def run_moving_window_decoding_ler_null(data1, data2, labels1, variables, time_window, n_reps=100, if_xgen=True):


    if time_window is not None:
        shattering_dim_rnd = np.zeros((n_reps, 2, 31))
        decoding_rnd = np.zeros((n_reps, 2, 4))
        xgen_decoding_rnd = np.zeros((n_reps, 2, 4))

        for rep in range(n_reps):
            print("Rep: ", rep + 1, " of ", n_reps)
            data_epochs1and4, labels_epochs1and4 = shuffle_stages(data1, data2, labels1)
            if if_xgen:
                shattering_dim_rep, decoding_rep, xgen_decoding_rep = run_moving_window_decoding(data_epochs1and4,
                                                                                                 labels_epochs1and4,
                                                                                                 variables, time_window,
                                                                                                 if_xgen=if_xgen)
                xgen_decoding_rnd[rep, :, :] = xgen_decoding_rep
            else:
                shattering_dim_rep, decoding_rep = run_moving_window_decoding(data_epochs1and4, labels_epochs1and4,
                                                                              variables, time_window, if_xgen=if_xgen)

            shattering_dim_rnd[rep, :, :] = shattering_dim_rep
            decoding_rnd[rep, :, :] = decoding_rep

        if if_xgen:
            return shattering_dim_rnd, decoding_rnd, xgen_decoding_rnd
        else:
            return shattering_dim_rnd, decoding_rnd
    if time_window is None:
        n_times = data1[0].shape[-1] - 80
        shattering_dim_rnd = np.zeros((n_times, n_reps, 2, 31))
        decoding_rnd = np.zeros((n_times, n_reps, 2, 4))
        for i_rep in range(n_reps):
            print("Rep: ", i_rep + 1, " of ", n_reps)
            data_epochs1and4, labels_epochs1and4 = shuffle_stages(data1, data2, labels1)
            for i_time in range(30, 200):
                time_sliding = [i_time, i_time + 1]
                print('Time point: ', i_time + 1, ' of ', data1[0].shape[-1] - 50)

                shattering_dim_t, decoding_t = run_moving_window_decoding(data_epochs1and4, labels_epochs1and4,
                                                                              variables, time_sliding, if_xgen=False)

                shattering_dim_rnd[i_time-30, i_rep, :, :] = shattering_dim_t
                decoding_rnd[i_time-30, i_rep, :, :] = decoding_t

        return shattering_dim_rnd, decoding_rnd
def save_data(data_list, names_list, path):
    data_dict = {}
    for i, name in enumerate(names_list):
        data_dict[name] = data_list[i]
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)


def load_data(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def line_plot_timevar(gs, fig, x, y, color, xlabel, ylabel, title, ylim, xlim, xticks=None, xticklabels=None,
                      baseline_line=None,
                      patch_pars=None, if_title=True, if_sem=False, xaxese_booo=False):
    ax = fig.add_subplot(gs)
    [ax.plot(x, y.mean(0)[_, :], color=color[_]) for _ in range(y.shape[1])]
    sns.despine(top=True, right=True)
    # plot the shaded area using standard error of the mean
    if if_sem:
        [ax.fill_between(x, y.mean(0)[_] - y.std(0)[_] / np.sqrt(y.shape[0]),
                         y.mean(0)[_] + y.std(0)[_] / np.sqrt(y.shape[0]), color=color[_], alpha=0.3) for _ in
         range(y.shape[1])]
    ax.set_ylabel(ylabel)
    if xaxese_booo:
        ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if if_title:
        ax.set_title(title)

    if xticks is not None:
        ax.set_xticks(xticks)
        [ax.axvline(xticks[_], linewidth=0.8, color='black', linestyle='--') for _ in range(len(xticks))]
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if baseline_line is not None:
        ax.axhline(baseline_line, color='black', linestyle='--', linewidth=0.8)

    if patch_pars is not None:
        rect = patches.Rectangle(patch_pars['xy'], patch_pars['width'], patch_pars['height'], edgecolor=None,
                                 facecolor='peachpuff',
                                 zorder=-8)
        ax.add_patch(rect)

    return ax


def plot_sig_bars(ax, dat_obs, dat_rnd, times, tails_lis, colour_lis, p_threshold=0.05, plot_chance_lvl=0.5,
                  if_smooth=False, variable_name='[name here]'):
    clt_times, clt_labels = compute_perm_stats(dat_obs,
                                               dat_rnd,
                                               tails=tails_lis,
                                               if_smooth=if_smooth
                                               )

    print('Cluster perm. test: ' + variable_name)
    for p in range(dat_obs.shape[0]):
        slices = get_slices(clt_times[p], clt_labels[p], times)
        for s_i in range(slices.shape[0]):
            if slices[s_i, 0] <= p_threshold:
                start_time = slices[s_i, 1]
                end_time = slices[s_i, 2]
                p_value = slices[s_i, 0]
                # Print time window for each significant cluster
                print(
                    f"   significant cluster found: {start_time:.3f}s to {end_time:.3f}s (duration: {end_time - start_time:.3f}s, p={p_value:.3f})")

                if p == 2:
                    ax.hlines(xmin=start_time, xmax=end_time, colors=colour_lis[p],
                              y=plot_chance_lvl - (p / 80), linestyles=(0, (1, 0.5)),
                              linewidth=2)
                else:
                    ax.hlines(xmin=start_time, xmax=end_time, colors=colour_lis[p],
                              y=plot_chance_lvl - (p / 80),
                              linewidth=2)

    return

def plot_scatter(gs, fig, x, y, scale, yaxis_label, xaxis_label, offset_axis=20, overlay_model=None, title=None,
                 plot_reg=False):
    ax = fig.add_subplot(gs)
    ax.scatter(x, y, color='darkgrey', s=10, zorder=-5,
               edgecolor='dimgrey')
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
    ax.set_xlabel(xaxis_label, labelpad=offset_axis)
    ax.set_ylabel(yaxis_label, labelpad=offset_axis)
    ax.set_title(title)

    if overlay_model == 'random':
        m_rnd = np.diag(np.cov(np.array([x, y]))).mean()
        n_circles = 4
        rnd_std = m_rnd ** 0.5
        r_all = rnd_std * 3
        r_range = np.linspace(0.0, r_all, n_circles)
        for c in range(n_circles):
            circle = plt.Circle((0, 0), r_range[c], color=sns.diverging_palette(230, 20, n=4)[-1], fill=False,
                                zorder=-4)
            ax.add_patch(circle)

    elif overlay_model == 'minimal1':
        ax.scatter(x=[0.0], y=[0.0], color=sns.diverging_palette(230, 20, n=4)[0], s=10, zorder=5)
    elif overlay_model == 'minimal2':
        m_opt = np.cov(np.array([x, y]))[1, 1]
        rnd_opt_all = (m_opt ** 0.5) * 3
        y_lis = np.linspace(-rnd_opt_all, rnd_opt_all, 40)
        x_lis = np.zeros(len(y_lis))
        ax.plot(x_lis, y_lis, color=sns.diverging_palette(230, 20, n=4)[0], linewidth=3, zorder=5)

    if plot_reg:
        b = np.linalg.lstsq(x[:, None], y[:, None])[0]
        x_line = np.linspace(-scale, scale, 100)
        ax.plot(x_line, b[0] * x_line, c='black', linewidth=0.8)

        corr = np.corrcoef(x, y)[0, 1]
        ax.text(scale * 1, scale * 0.7, 'r=' + str(round(corr, 2)), fontsize=11, color='black', ha='center',
                va='center')

    return


def plot_cov(gs, fig, dat, scale=0.007):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = fig.add_subplot(gs)
    norm = colors.TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    im = ax.imshow(np.cov(dat.T), cmap=sns.diverging_palette(230, 20, as_cmap=True), norm=norm,
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
    return


def compute_dist_random(selectivity_coeffs_stages, n_bootstraps=1000, rnd_model='gaussian (spherical)',
                        design_model='0/1', bon_correction=False, metric='KL divergance estimate', relative_dist=True):
    n_stages = len(selectivity_coeffs_stages)
    p_vals = []

    KL = np.zeros((n_stages, n_bootstraps))
    KL_r = np.zeros((n_stages, n_bootstraps))
    KL_opt = np.zeros((n_stages, n_bootstraps))
    for i_stage in range(n_stages):
        for bootstrap in tqdm(range(n_bootstraps)):
            data_train = selectivity_coeffs_stages[i_stage][:, :, 0]
            data_test = selectivity_coeffs_stages[i_stage][:, :, 1]

            if design_model == '0/1':
                opt_cov = np.zeros((3, 3))
                m = np.mean(np.diag(np.cov(data_train.T)))
                opt_cov[:2, :2] = 0.5 * m
                opt_cov[2, 2] = 2 * m
                opt_cov[2, :2] = -m
                opt_cov[:2, 2] = -m

            elif design_model == '+1/-1':
                # m = np.mean(np.diag(np.cov(data_train.T)))
                # opt_cov = np.diag([0, 0, m * 3])
                opt_cov = np.diag([0, 0, np.cov(data_train.T)[2, 2]])

            s_opt = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                  opt_cov,
                                                  data_train.shape[0])

            if rnd_model == 'gaussian (spherical)':
                m = np.mean(np.diag(np.cov(data_train.T)))
                shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                           np.diag([m, m, m]),
                                                           data_train.shape[0])
                m = np.mean(np.diag(np.cov(data_test.T)))
                shuffled_2 = np.random.multivariate_normal(np.zeros(data_test.shape[1]),
                                                           np.diag([m, m, m]),
                                                           data_test.shape[0])
                if metric == 'KL divergance estimate':
                    kl_itr = 0.5 * (KLdivergence(data_test, shuffled_1) + KLdivergence(shuffled_1, data_test))
                    kl_r_itr = 0.5 * (KLdivergence(shuffled_2, shuffled_1) + KLdivergence(shuffled_1, shuffled_2))
                elif metric == 'euclidean distance':
                    kl_itr = euclidean_distance(data_test, shuffled_1)
                    kl_r_itr = euclidean_distance(shuffled_2, shuffled_1)
                elif metric == 'epairs':
                    kl_itr = epairs_metric(data_test, shuffled_1)
                    kl_r_itr = epairs_metric(shuffled_2, shuffled_1)

            elif rnd_model == 'gaussian (tied)':

                shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                           np.diag(np.diag(np.cov(data_train.T))),
                                                           data_train.shape[0])

                shuffled_2 = np.random.multivariate_normal(np.zeros(data_test.shape[1]),
                                                           np.diag(np.diag(np.cov(data_test.T))),
                                                           data_test.shape[0])

                if metric == 'KL divergance estimate':
                    kl_itr = 0.5 * (KLdivergence(data_test, shuffled_1) + KLdivergence(shuffled_1, data_test))
                    kl_r_itr = 0.5 * (KLdivergence(shuffled_2, shuffled_1) + KLdivergence(shuffled_1, shuffled_2))
                elif metric == 'euclidean distance':
                    kl_itr = euclidean_distance(data_test, shuffled_1)
                    kl_r_itr = euclidean_distance(shuffled_2, shuffled_1)
                elif metric == 'epairs':
                    kl_itr = epairs_metric(data_test, shuffled_1)
                    kl_r_itr = epairs_metric(shuffled_2, shuffled_1)

            KL[i_stage, bootstrap] = kl_itr
            KL_r[i_stage, bootstrap] = kl_r_itr

            if metric == 'KL divergance estimate':
                KL_opt[i_stage, bootstrap] = 0.5 * (KLdivergence(s_opt, shuffled_1) + KLdivergence(shuffled_1, s_opt))
            elif metric == 'euclidean distance':
                KL_opt[i_stage, bootstrap] = euclidean_distance(s_opt, shuffled_1)
            elif metric == 'epairs':
                KL_opt[i_stage, bootstrap] = epairs_metric(s_opt, shuffled_1)

    if relative_dist:
        KL_r_avg = np.mean(KL_r, keepdims=True, axis=-1)
        KL -= KL_r_avg
        KL_r -= KL_r_avg
        KL_opt -= KL_r_avg
        KL_opt_avg = np.mean(KL_opt, keepdims=True, axis=-1)
        KL /= KL_opt_avg
        KL_r /= KL_opt_avg
        KL_opt /= KL_opt_avg

    p = 1 * (np.sum(KL_r >= np.mean(KL, axis=-1, keepdims=True), axis=-1) / n_bootstraps)
    if bon_correction:
        p = p * n_stages
    p_vals.append(p)

    print(p_vals)
    return [KL, KL_r, KL_opt], p_vals


def compute_dist_structured(selectivity_coeffs_stages, n_bootstraps=1000, rnd_model='gaussian (spherical)',
                            design_model='0/1', bon_correction=False, metric='KL divergance estimate',
                            relative_dist=True):
    n_epochs = len(selectivity_coeffs_stages)
    p_vals = []
    KL = np.zeros((n_epochs, n_bootstraps))
    KL_r = np.zeros((n_epochs, n_bootstraps))
    KL_opt = np.zeros((n_epochs, n_bootstraps))
    for e in range(n_epochs):
        for bootstrap in tqdm(range(n_bootstraps)):

            data_train = selectivity_coeffs_stages[e][:, :, 0]
            data_test = selectivity_coeffs_stages[e][:, :, 1]

            if design_model == '0/1':

                opt_cov1 = np.zeros((3, 3))
                m = np.mean(np.diag(np.cov(data_train.T)))
                opt_cov1[:2, :2] = 0.5 * m
                opt_cov1[2, 2] = 2 * m
                opt_cov1[2, :2] = -m;
                opt_cov1[:2, 2] = -m

                opt_cov2 = np.zeros((3, 3))
                m = np.mean(np.diag(np.cov(data_test.T)))
                opt_cov2[:2, :2] = 0.5 * m
                opt_cov2[2, 2] = 2 * m
                opt_cov2[2, :2] = -m;
                opt_cov2[:2, 2] = -m

            elif design_model == '+1/-1':

                opt_cov1 = np.diag([0, 0, np.cov(data_train.T)[2, 2]])
                opt_cov2 = np.diag([0, 0, np.cov(data_test.T)[2, 2]])

            s_opt_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                    opt_cov1,
                                                    data_train.shape[0])

            s_opt_2 = np.random.multivariate_normal(np.zeros(data_test.shape[1]),
                                                    opt_cov2,
                                                    data_test.shape[0])

            if rnd_model == 'gaussian (tied)':
                shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                           np.diag(np.diag(np.cov(data_train.T))),
                                                           data_train.shape[0])

                if metric == 'KL divergance estimate':
                    KL[e, bootstrap] = 0.5 * (KLdivergence(data_test, s_opt_1) + KLdivergence(s_opt_1, data_test))
                    KL_r[e, bootstrap] = 0.5 * (
                            KLdivergence(shuffled_1, s_opt_1) + KLdivergence(s_opt_1, shuffled_1))
                    KL_opt[e, bootstrap] = 0.5 * (KLdivergence(s_opt_2, s_opt_1) + KLdivergence(s_opt_1, s_opt_2))
                elif metric == 'euclidean distance':
                    KL[e, bootstrap] = euclidean_distance(data_test, s_opt_1)
                    KL_r[e, bootstrap] = euclidean_distance(shuffled_1, s_opt_1)
                    KL_opt[e, bootstrap] = euclidean_distance(s_opt_2, s_opt_1)
                elif metric == 'epairs':
                    KL[e, bootstrap] = epairs_metric(data_test, s_opt_1)
                    KL_r[e, bootstrap] = epairs_metric(shuffled_1, s_opt_1)
                    KL_opt[e, bootstrap] = epairs_metric(s_opt_2, s_opt_1)

            if rnd_model == 'gaussian (spherical)':
                m = np.mean(np.diag(np.cov(data_train.T)))
                shuffled_1 = np.random.multivariate_normal(np.zeros(data_train.shape[1]),
                                                           np.diag([m, m, m]),
                                                           data_train.shape[0])
                if metric == 'KL divergance estimate':
                    KL[e, bootstrap] = 0.5 * (KLdivergence(data_test, s_opt_1) + KLdivergence(s_opt_1, data_test))
                    KL_r[e, bootstrap] = 0.5 * (
                            KLdivergence(shuffled_1, s_opt_1) + KLdivergence(s_opt_1, shuffled_1))
                    KL_opt[e, bootstrap] = 0.5 * (KLdivergence(s_opt_2, s_opt_1) + KLdivergence(s_opt_1, s_opt_2))

                elif metric == 'euclidean distance':
                    KL[e, bootstrap] = euclidean_distance(data_test, s_opt_1)
                    KL_r[e, bootstrap] = euclidean_distance(shuffled_1, s_opt_1)
                    KL_opt[e, bootstrap] = euclidean_distance(s_opt_2, s_opt_1)

                elif metric == 'epairs':
                    KL[e, bootstrap] = epairs_metric(data_test, s_opt_1)
                    KL_r[e, bootstrap] = epairs_metric(shuffled_1, s_opt_1)
                    KL_opt[e, bootstrap] = epairs_metric(s_opt_2, s_opt_1)

    if relative_dist:
        KL_opt_avg = np.mean(KL_opt, keepdims=True, axis=-1)
        KL -= KL_opt_avg
        KL_r -= KL_opt_avg
        KL_opt -= KL_opt_avg
        KL_r_avg = np.mean(KL_r, keepdims=True, axis=-1)
        KL /= KL_r_avg
        KL_r /= KL_r_avg
        KL_opt /= KL_r_avg

    p = 1 * (np.sum(KL_r <= np.mean(KL, axis=-1, keepdims=True), axis=-1) / n_bootstraps)
    if bon_correction:
        p = p * n_epochs

    p_vals.append(p)

    return [KL, KL_r, KL_opt], p_vals


def plot_distance(gs, fig, dat_matrix, title, pvals=None, dist_dot=30, x_stages_ticks=None, x_label='learning stage'):
    col_data = colors.to_rgb('black')
    col_rnd = sns.diverging_palette(230, 20, n=4)[-1]
    col_opt = sns.diverging_palette(230, 20, n=4)[0]

    n_stages = dat_matrix[0].shape[0]

    x_stages = list(range(1, n_stages + 1))
    ax = fig.add_subplot(gs)
    ax.scatter(x_stages, dat_matrix[0].mean(-1), color=col_data, s=dist_dot, zorder=5)
    ax.scatter(x_stages, dat_matrix[1].mean(-1), color=col_rnd, s=dist_dot)
    ax.scatter(x_stages, dat_matrix[2].mean(-1), color=col_opt, s=dist_dot)
    ax.errorbar(x_stages, dat_matrix[0].mean(-1), fmt='', yerr=dat_matrix[0].std(-1), color=col_data,
                alpha=0.5, zorder=5)
    ax.errorbar(x_stages, dat_matrix[1].mean(-1), fmt='', yerr=dat_matrix[1].std(-1), color=col_rnd,
                alpha=0.5)
    ax.errorbar(x_stages, dat_matrix[2].mean(-1), fmt='', yerr=dat_matrix[2].std(-1), color=col_opt,
                alpha=0.5)

    # plot p-vales for each learning stage
    if pvals is not None:
        for i_stage in range(n_stages):
            ax.text(i_stage + 1, 0.5, p_into_stars(pvals[i_stage]), fontsize=10, color='black', ha='center',
                    va='center')
    if x_stages_ticks is None:
        ax.set_xticks(x_stages)
        ax.set_xticklabels(x_stages)
    else:
        ax.set_xticks(x_stages)
        ax.set_xticklabels(x_stages_ticks)
    ax.set_xlabel(x_label)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    sns.despine(right=True, top=True)
    ax.set_ylabel('relative\neuclidean dist.')
    ax.set_title(title)

    mean_dat_distances = dat_matrix[0].mean(-1)
    print(title)
    for _ in range(n_stages):
        print('    Stage ', str(_+1), '   M = ', str(round(mean_dat_distances[_], 3)), ', p-value = ', str(round(pvals[_], 3)))


    return


def plot_decoding(gs, fig, obs_decoding, rnd_decoding, rnd_ler_decoding, title, y_lim,
                  tails_lis=['smaller', 'smaller', 'greater']):
    # get rid of the width column
    obs_decoding = np.concatenate([obs_decoding[:, :2], obs_decoding[:, 3][:, None]], axis=1)
    rnd_ler_decoding = np.concatenate([rnd_ler_decoding[:, :, :2], rnd_ler_decoding[:, :, 3][:, :, None]], axis=2)

    x_data = np.arange(3)
    ax = fig.add_subplot(gs)
    ax.plot(x_data, obs_decoding[0, :], color='grey')
    ax.scatter(x_data, obs_decoding[0, :], color='grey')
    ax.plot(x_data, obs_decoding[-1, :], color='black', zorder=5)
    ax.scatter(x_data, obs_decoding[-1, :], color='black', zorder=5)

    ax.set_xticks(x_data)
    ax.set_xticklabels(['colour', 'shape', 'xor'])
    plt.margins(x=0.1)
    sns.despine(right=True, top=True)
    ax.set_ylabel('accuracy')
    ax.set_ylim(y_lim)
    ax.set_title(title)

    rand_null = rnd_decoding.flatten()
    dec_mean = rand_null.mean()
    dec_std = rand_null.std()
    # ax.axhline(dec_mean, linestyle='-', linewidth=1, color=col_chance)
    ax.axhline(dec_mean + 2 * dec_std, linestyle='--', linewidth=1, color='grey')
    ax.axhline(dec_mean - 2 * dec_std, linestyle='--', linewidth=1, color='grey')
    rect = patches.Rectangle((-0.4, dec_mean - 2 * dec_std), 5.5, 4 * dec_std, edgecolor=None, facecolor='lightgrey',
                             zorder=-8)

    ax.add_patch(rect)

    for i_var in range(len(tails_lis)):
        p_value = compute_p_value(obs_decoding[-1, i_var], obs_decoding[0, i_var],
                                  rnd_ler_decoding[:, -1, i_var], rnd_ler_decoding[:, 0, i_var],
                                  tail=tails_lis[i_var])
        print(p_value)

        # plot the p-values as stars
        ax.text(i_var, 0.8, p_into_stars(p_value), fontsize=10, color='black', ha='center', va='center')

    return


def compute_firing_rate_norm(dat):
    return np.array([np.linalg.norm(dat[:, i_time]) / np.sqrt(dat.shape[0]) for i_time in range(dat.shape[-1])])


def compute_f_rate_norm_stages(data_eq, labels_eq, baseline=True, if_avg=False):
    n_stages = len(data_eq)
    n_windows = data_eq[0].shape[0]
    f_rates_stage = np.zeros((n_stages, data_eq[0].shape[-1], n_windows))
    for i_win in range(n_windows):
        for i_stage in range(n_stages):
            data_stage = data_eq[i_stage][i_win, :, :, :]
            labels_stage = labels_eq[i_stage][i_win, :]

            if if_avg:
                data_stage = condi_avg(data_stage, labels_stage)
            data_stage_collapsed = data_stage.reshape(-1, data_stage.shape[-1])
            data_stage_norm = compute_firing_rate_norm(data_stage_collapsed)
            f_rates_stage[i_stage, :, i_win] = data_stage_norm

    f_rates_stages = f_rates_stage.mean(-1)

    if baseline:
        f_rates_stages = f_rates_stages - np.mean(f_rates_stages[:, :50], axis=-1, keepdims=True)

    return f_rates_stages


def compute_f_rate_norm_ler_null(data1, data2, labels1, n_reps=100, baseline=True):
    f_rate_ler_null = np.zeros((n_reps, 2, data1.shape[-1]))
    for i_rep in range(n_reps):
        print("Rep: ", i_rep + 1, " of ", n_reps)
        data_epochs1and4, labels_epochs1and4 = shuffle_stages(data1, data2, labels1)
        f_rate_ler_null[i_rep, :, :] = compute_f_rate_norm_stages(data_epochs1and4, labels_epochs1and4,
                                                                  baseline=baseline)

    return f_rate_ler_null


def plot_regression_sup(x, y, gs, fig, color='black', xlabel=None, ylabel=None, title=None, compute_r=True,
                        tail='two-sided'):
    ax = fig.add_subplot(gs)
    sns.regplot(x=x, y=y, ax=ax, color=color, scatter_kws={'s': 20})
    if compute_r:
        rho, pval = stats.spearmanr(x, y, alternative=tail)
        ax.text((np.max(x) + np.min(x)) / 2, np.max(y) - 0.01 * np.max(y),
                'rho = {0:.2f}\np = {1:.2f}'.format(rho, pval), color='black', fontsize=10, ha='center')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sns.despine(top=True, right=True)


def run_time_resolved_decoding(data, labels, if_save=True, fname='exp1_decoding_time', mode='within_stage', time_resolved=False):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    variables = [np.array(config['ENCODING_EXP1']['colour']),
                 np.array(config['ENCODING_EXP1']['shape']),
                 np.array(config['ENCODING_EXP1']['width']),
                 np.array(config['ENCODING_EXP1']['xor'])]

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=98)
    data_eq = [data_eq[0], data_eq[-1]]
    labels_eq = [labels_eq[0], labels_eq[-1]]

    if (mode == 'only_ler_null') & (time_resolved==False):
        shattering_dim_rnd_time = []
        decoding_rnd_time = []
        for i_time in range(30, 200):
            time_sliding = [i_time, i_time + 1]
            print('Time point: ', i_time + 1, ' of ', data_eq[0].shape[-1] - 50)
            shattering_dim_rnd, decoding_rnd = run_moving_window_decoding_ler_null(data_eq[0], data_eq[-1],
                                                                                   labels_eq[0],
                                                                                   variables, time_window=time_sliding,
                                                                                   n_reps=config['ANALYSIS_PARAMS'][
                                                                                       'N_REPS'], if_xgen=False)
            shattering_dim_rnd_time.append(shattering_dim_rnd)
            decoding_rnd_time.append(decoding_rnd)

        shattering_dim_rnd_time = np.array(shattering_dim_rnd_time)
        decoding_rnd_time = np.array(decoding_rnd_time)

        if if_save:
            data_exp1 = [shattering_dim_rnd_time, decoding_rnd_time]
            names_exp1 = ['shattering_dim_rnd_time', 'decoding_rnd_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_ler_null.pickle')

        return shattering_dim_rnd_time, decoding_rnd_time

    if (mode == 'only_ler_null') & (time_resolved==True):

        shattering_dim_rnd_time, decoding_rnd_time = run_moving_window_decoding_ler_null(data_eq[0], data_eq[-1],
                                                                               labels_eq[0],
                                                                               variables, time_window=None,
                                                                               n_reps=config['ANALYSIS_PARAMS'][
                                                                                   'N_REPS'], if_xgen=False)

        if if_save:
            data_exp1 = [shattering_dim_rnd_time, decoding_rnd_time]
            names_exp1 = ['shattering_dim_rnd_time', 'decoding_rnd_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_ler_null.pickle')

        return shattering_dim_rnd_time, decoding_rnd_time



    elif mode == 'within_stage':
        shattering_dim_time = []
        decoding_time = []
        decoding_null_time = []
        for i_time in range(30, 200):
            time_sliding = [i_time, i_time + 1]
            print('Time point: ', i_time + 1, ' of ', data_eq[0].shape[-1] - 50)
            shattering_dim, decoding, decoding_null = run_moving_window_decoding(data_eq, labels_eq, variables,
                                                                                 time_window=time_sliding, method='svm',
                                                                                 if_xgen=False, if_null=True,
                                                                                 n_reps=config['ANALYSIS_PARAMS'][
                                                                                     'N_REPS'])

            shattering_dim_time.append(shattering_dim)
            decoding_time.append(decoding)
            decoding_null_time.append(decoding_null)

        shattering_dim_time = np.array(shattering_dim_time)
        decoding_time = np.array(decoding_time)
        decoding_null_time = np.array(decoding_null_time)

        if if_save:
            data_exp1 = [shattering_dim_time, decoding_time, decoding_null_time]
            names_exp1 = ['shattering_dim_time', 'decoding_time', 'decoding_null_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_within_stage.pickle')

        return shattering_dim_time, decoding_time, decoding_null_time


def compute_lifetime_sparseness(data):
    l_sparse = (1 - (np.mean(data, axis=0) ** 2) / np.mean(data ** 2, axis=0)) / (1 - 1 / data.shape[1])
    return l_sparse


def plot_sparseness(gs, fig, data1, data2):
    ax = fig.add_subplot(gs)
    ax.hist(data1, color='grey')
    ax.hist(data2, color='black')
    sns.despine(top=True, right=True)
    ax.set_ylabel('neurons (n)')
    ax.set_xlabel('lifetime sparseness')
    ax.set_title('sparse coding\n(stage 1 vs stage 4)')
    from scipy import stats
    D, sparse_p_value = stats.kstest(data1, data2)
    print( 'D = ', str(round(D,3)), ', p-value = ', str(round(sparse_p_value,3)))
    ax.annotate(p_into_stars(sparse_p_value), xy=(.6, 20), ha='center', size=10)


def plot_pca(gs, fig, obs_pca, rnd_pca, title, y_lim):
    ax = fig.add_subplot(gs)
    x_data = list(range(1, obs_pca.shape[0] + 1))
    ax.plot(x_data, obs_pca[:, 0], color='black', zorder=5)
    ax.scatter(x_data, obs_pca[:, 0], color='black', zorder=5)

    ax.set_xticks(x_data)
    ax.set_xticklabels(x_data)
    plt.margins(x=0.1)
    sns.despine(right=True, top=True)
    ax.set_ylabel('var. explained')
    ax.set_xlabel('learning stage')
    ax.set_ylim(y_lim)
    ax.set_title(title)

    p_value = compute_p_value(obs_pca[-1, 0], obs_pca[0, 0], rnd_pca[:, -1, 0], rnd_pca[:, 0, 0], tail='greater')
    ax.text(2.5, 0.55, p_into_stars(p_value), fontsize=10, color='black', ha='center', va='center')


def run_pca_movewin(data, labels, n_comps=4, factor=[0, 0, 1, 1, 2, 2, 3, 3], n_reps=100):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    min_cell = np.min([data[i_stage].shape[2] for i_stage in range(len(data))])


    pc_var = np.zeros((len(data), 4, config['ANALYSIS_PARAMS']['N_WINDOWS'], n_reps))
    for i_stage in range(len(data)):
        data_re = data[i_stage][:, :, :,
                       config['ANALYSIS_PARAMS']['TIME_WINDOW'][0]:config['ANALYSIS_PARAMS']['TIME_WINDOW'][1]]
        for i_rep in range(n_reps):
            idx_rnd = np.arange(data_re.shape[2])
            idx_rnd = np.random.choice(idx_rnd, min_cell, replace=False)
            data_re = data_re[:, :, idx_rnd,:]
            for i_win in range(config['ANALYSIS_PARAMS']['N_WINDOWS']):
                data_avg = data_re[i_win, :, :, :]
                pc_var[i_stage, :, i_win, i_rep] = cross_val_pca(data=data_avg, labels=labels[i_stage][i_win, :],
                                                          n_comps=n_comps, factor=factor)

    pc_var = pc_var.mean(-1)

    return pc_var.mean(-1)


def run_pca_movewin_null(data1, data2, labels1, n_comps=4, factor=[0, 0, 1, 1, 2, 2, 3, 3]):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    pc_var_rnd = []
    for i_rep in range(config['ANALYSIS_PARAMS']['N_REPS']):
        print("Rep: ", i_rep + 1, " of ", config['ANALYSIS_PARAMS']['N_REPS'])
        data_epochs1and4, labels_epochs1and4 = shuffle_stages(data1, data2, labels1)
        pc_var_rnd.append(run_pca_movewin(data_epochs1and4, labels_epochs1and4, n_comps=n_comps, factor=factor))

    return np.array(pc_var_rnd)


def plot_lin_nonlin_dims(gs, fig, lin_obs, nonlin_obs, lin_rnd, nonlin_rnd, y_lim=[0.5, 0.8],
                         title='types of computations'):
    ax = fig.add_subplot(gs)
    x_data = list(range(1, lin_obs.shape[0] + 1))
    ax.scatter(x_data, lin_obs.mean(-1), color='grey', zorder=-5)
    ax.errorbar(x_data, lin_obs.mean(-1), yerr=lin_obs.std(-1), color='grey', zorder=-5)

    ax.scatter(x_data, nonlin_obs.mean(-1), color='black', zorder=5)
    ax.errorbar(x_data, nonlin_obs.mean(-1), yerr=nonlin_obs.std(-1), color='black', zorder=5)

    ax.set_xticks(x_data)
    ax.set_xticklabels(x_data)
    plt.margins(x=0.1)
    sns.despine(right=True, top=True)
    ax.set_ylabel('accuracy')
    ax.set_xlabel('learning stage')
    ax.set_ylim(y_lim)
    ax.set_title(title)
    print ("  Linear dimensionality")
    p_value_lin = compute_p_value(lin_obs.mean(-1)[-1], lin_obs.mean(-1)[0], lin_rnd.mean(-1)[:, -1],
                                  lin_rnd.mean(-1)[:, 0], tail='smaller')
    ax.text(2.5, 0.72, p_into_stars(p_value_lin), fontsize=10, color='grey', ha='center', va='center')

    print("  Nonlinear dimensionality")
    p_value_nonlin = compute_p_value(nonlin_obs.mean(-1)[-1], nonlin_obs.mean(-1)[0], nonlin_rnd.mean(-1)[:, -1],
                                  nonlin_rnd.mean(-1)[:, 0], tail='two')
    ax.text(2.5, 0.60, p_into_stars(p_value_nonlin), fontsize=10, color='black', ha='center', va='center')

    return


def run_moving_window_xgen(data_eq, labels_eq, variables, time_window, method='svm', n_jobs=None,
                           if_null=False, n_reps=None):
    n_windows = data_eq[0].shape[0]
    n_stages = len(data_eq)
    xgen_decoding = np.zeros((n_stages, 4, n_windows))

    for i_stage in range(n_stages):
        for i_window in range(n_windows):
            X_stage = data_eq[i_stage][i_window, :, :, :]
            y_stage = labels_eq[i_stage][i_window, :]

            decoding_xgen_sess = get_xgen(data=X_stage,
                                          labels=y_stage,
                                          variables=variables,
                                          time_window=time_window,
                                          method=method,
                                          mode='only_rel',
                                          n_jobs=n_jobs,
                                          )
            xgen_decoding[i_stage, :, i_window] = decoding_xgen_sess

    xgen_decoding = xgen_decoding.mean(-1)

    if if_null:
        xgen_null = np.zeros((n_stages, n_reps, 4, n_windows))
        for i_rep in tqdm(range(n_reps)):
            for i_stage in range(n_stages):
                X_stage = data_eq[i_stage]
                y_stage = labels_eq[i_stage]

                idc_rnd = np.arange(X_stage.shape[1])
                random.shuffle(idc_rnd)
                y_stage_rnd = y_stage[:, idc_rnd]

                for i_window in range(n_windows):
                    X_stage_win = X_stage[i_window, :, :, :]
                    y_stage_win = y_stage_rnd[i_window, :]

                    xgen_null[i_stage, i_rep, :, i_window] = get_xgen(data=X_stage_win,
                                                                      labels=y_stage_win,
                                                                      variables=variables,
                                                                      time_window=time_window,
                                                                      method=method,
                                                                      mode='only_rel',
                                                                      n_jobs=n_jobs,
                                                                      verbose=False
                                                                      )

        xgen_null = xgen_null.mean(-1)

    if if_null:
        return xgen_decoding, xgen_null
    else:
        return xgen_decoding


def run_moving_window_xgen_ler_null(data1, data2, labels1, variables, time_window, n_reps=100):

    if time_window is not None:
        xgen_decoding_rnd = np.zeros((n_reps, 2, 4))
        for rep in range(n_reps):
            print("Rep: ", rep + 1, " of ", n_reps)
            data_epochs1and4, labels_epochs1and4 = shuffle_stages(data1, data2, labels1)
            xgen_decoding_rnd[rep, :, :] = run_moving_window_xgen(data_epochs1and4, labels_epochs1and4, variables,
                                                                  time_window)

    if time_window is None:
        n_times = data1[0].shape[-1] - 80
        xgen_decoding_rnd = np.zeros((n_times, n_reps, 2, 4))
        for i_rep in range(n_reps):
            print("Rep: ", i_rep + 1, " of ", n_reps)
            data_epochs1and4, labels_epochs1and4 = shuffle_stages(data1, data2, labels1)
            for i_time in range(30, 200):
                time_sliding = [i_time, i_time + 1]
                print('Time point: ', i_time + 1, ' of ', data1[0].shape[-1] - 50)

                xgen_dec_t = run_moving_window_xgen(data_epochs1and4, labels_epochs1and4, variables,
                                                                      time_sliding)


                xgen_decoding_rnd[i_time-30, i_rep, :, :] = xgen_dec_t


    return xgen_decoding_rnd


def run_time_resolved_xgen(data, labels, if_save=True, fname='exp1_xgen_time', mode='within_stage', time_resolved=False):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    variables = [np.array(config['ENCODING_EXP1']['colour']),
                 np.array(config['ENCODING_EXP1']['shape']),
                 np.array(config['ENCODING_EXP1']['width']),
                 np.array(config['ENCODING_EXP1']['xor'])]

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=98)
    data_eq = [data_eq[0], data_eq[-1]]
    labels_eq = [labels_eq[0], labels_eq[-1]]



    if mode == 'only_ler_null':

        xgen_rnd_time = run_moving_window_xgen_ler_null(data_eq[0], data_eq[-1], labels_eq[0],
                                                   variables, time_window=None,
                                                   n_reps=config['ANALYSIS_PARAMS'][
                                                       'N_REPS'])


        if if_save:
            data_exp1 = [xgen_rnd_time]
            names_exp1 = ['xgen_rnd_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_ler_null.pickle')

        return xgen_rnd_time


    elif mode == 'within_stage':
        xgen_time = []
        xgen_time_null = []
        for i_time in range(30, 200):
            time_sliding = [i_time, i_time + 1]
            print('Time point: ', i_time + 1, ' of ', data_eq[0].shape[-1] - 50)
            xgen, xgen_null = run_moving_window_xgen(data_eq, labels_eq, variables,
                                                     time_window=time_sliding, method='svm',
                                                     if_null=True,
                                                     n_reps=config['ANALYSIS_PARAMS']['N_REPS'])

            xgen_time.append(xgen)
            xgen_time_null.append(xgen_null)

        xgen_time_null = np.array(xgen_time_null)
        xgen_time = np.array(xgen_time)

        if if_save:
            data_exp1 = [xgen_time, xgen_time_null]
            names_exp1 = ['xgen_time', 'xgen_time_null']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_within_stage.pickle')

        return xgen_time, xgen_time_null

    elif mode == 'within_stage_without_null':
        xgen_time = []
        for i_time in range(30, 200):
            time_sliding = [i_time, i_time + 1]
            print('Time point: ', i_time + 1, ' of ', data_eq[0].shape[-1] - 50)
            xgen = run_moving_window_xgen(data_eq, labels_eq, variables,
                                          time_window=time_sliding, method='svm',
                                          n_reps=config['ANALYSIS_PARAMS']['N_REPS'])

            xgen_time.append(xgen)

        xgen_time = np.array(xgen_time)

        if if_save:
            data_exp1 = [xgen_time]
            names_exp1 = ['xgen_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_within_stage.pickle')

        return xgen_time


def split_data_dec(X, y, y_splitting):
    X1 = X[y_splitting == 0, :]
    X2 = X[y_splitting == 1, :]
    y1 = y[y_splitting == 0]
    y2 = y[y_splitting == 1]
    return X1, X2, y1, y2


def run_decoding_colour_locked(data_wins, labels_wins, window_early, factors, mode='only_observed'):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    if mode == 'only_observed':
        n_stages = len(data_wins)
        scores_context = np.zeros((2, n_stages, 3))
        scores_set = np.zeros((2, n_stages, 3))
        for i_stage in range(n_stages):
            for i_window in range(config['ANALYSIS_PARAMS']['N_WINDOWS']):
                X = np.mean(data_wins[i_stage][i_window, :, :, window_early[0]:window_early[1]], axis=-1)
                y = labels_wins[i_stage][i_window, :]

                y_context = np.array(assign_lables(y, factors[0]))
                y_set = np.array(assign_lables(y, factors[1]))
                scores_context[0, i_stage, i_window] = decode(X, y_context)
                scores_set[0, i_stage, i_window] = decode(X, y_set)

                X1_set1, X2_set2, y1_set1, y2_set2 = split_data_dec(X, y_context, y_splitting=y_set)
                scores_context[1, i_stage, i_window] = decode_xgen(X1_set1, X2_set2, y1_set1, y2_set2, n_jobs=None)

                X1_cxt1, X2_cxt2, y1_cxt1, y2_cxt2 = split_data_dec(X, y_set, y_splitting=y_context)
                scores_set[1, i_stage, i_window] = decode_xgen(X1_cxt1, X2_cxt2, y1_cxt1, y2_cxt2, n_jobs=None)

        return scores_context.mean(-1), scores_set.mean(-1)
    if mode == 'only_null':
        n_stages = len(data_wins)
        scores_rnd = np.zeros((2, n_stages, 3, config['ANALYSIS_PARAMS']['N_REPS']))
        for i_stage in range(n_stages):
            for i_rep in range(config['ANALYSIS_PARAMS']['N_REPS']):
                idc_rnd = np.arange(data_wins[i_stage].shape[1])
                random.shuffle(idc_rnd)
                for i_window in range(config['ANALYSIS_PARAMS']['N_WINDOWS']):
                    X = np.mean(data_wins[i_stage][i_window, :, :, window_early[0]:window_early[1]], axis=-1)
                    y = labels_wins[i_stage][i_window, :]

                    X = X[idc_rnd, :]
                    y_context = np.array(assign_lables(y, factors[0]))
                    y_set = np.array(assign_lables(y, factors[1]))
                    scores_rnd[0, i_stage, i_window, i_rep] = decode(X, y_context)

                    X1_set1, X2_set2, y1_set1, y2_set2 = split_data_dec(X, y_context, y_splitting=y_set)
                    scores_rnd[1, i_stage, i_window, i_rep] = decode_xgen(X1_set1, X2_set2, y1_set1, y2_set2, n_jobs=None)

        return scores_rnd.mean(-2)




def run_decoding_ler_null_colour_locked(data_wins, labels_wins, factors, time_window=[90, 100], n_reps=10):
    dat_stage1 = data_wins[0]
    dat_stage4 = data_wins[3]
    label_stage1 = labels_wins[0]
    label_stage4 = labels_wins[3]

    data_all = np.concatenate((dat_stage1, dat_stage4), axis=2)
    n_cells_stage1 = dat_stage1.shape[2]
    n_cells_stage4 = dat_stage4.shape[2]
    n_cells_all = data_all.shape[2]

    n_stages = len([dat_stage1, dat_stage4])
    scores_context_rnd = np.zeros((n_reps, 2, n_stages))
    scores_set_rnd = np.zeros((n_reps, 2, n_stages))
    for rep in range(n_reps):
        print("Rep: ", rep + 1, " of ", n_reps)
        idc_cells = np.arange(n_cells_all)
        random.shuffle(idc_cells)
        dat_stage1_shuffled = data_all[:, :, idc_cells[:n_cells_stage1], :]
        dat_stage4_shuffled = data_all[:, :, idc_cells[n_cells_stage1:], :]
        data_epochs1and4 = [dat_stage1_shuffled, dat_stage4_shuffled]
        labels_epochs1and4 = [label_stage1, label_stage4]
        scores_context_rnd_rep, scores_set_rnd_rep = run_decoding_colour_locked(data_epochs1and4,
                                                                                labels_epochs1and4,
                                                                                time_window,
                                                                                factors)

        scores_context_rnd[rep, :, :] = scores_context_rnd_rep
        scores_set_rnd[rep, :, :] = scores_set_rnd_rep

    return scores_context_rnd, scores_set_rnd


def run_decoding_shape_locked(data_wins, labels_wins, time_window, factors):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    n_stages = len(data_wins)
    scores_context = np.zeros((2, n_stages, config['ANALYSIS_PARAMS']['N_WINDOWS']))
    scores_set = np.zeros((2, n_stages, config['ANALYSIS_PARAMS']['N_WINDOWS']))
    scores_shape = np.zeros((2, n_stages, config['ANALYSIS_PARAMS']['N_WINDOWS']))
    scores_width = np.zeros((2, n_stages, config['ANALYSIS_PARAMS']['N_WINDOWS']))
    scores_xor = np.zeros((2, n_stages, config['ANALYSIS_PARAMS']['N_WINDOWS']))
    for i_stage in range(n_stages):
        for i_window in range(config['ANALYSIS_PARAMS']['N_WINDOWS']):
            X = np.mean(data_wins[i_stage][i_window, :, :, time_window[0]:time_window[1]], axis=-1)
            y = labels_wins[i_stage][i_window, :]
            y_context = np.array(assign_lables(y, factors[0]))
            y_set = np.array(assign_lables(y, factors[1]))
            y_shape = np.array(assign_lables(y, factors[2]))
            y_width = np.array(assign_lables(y, factors[3]))
            y_xor = np.array(assign_lables(y, factors[4]))

            scores_context[0, i_stage, i_window] = decode(X, y_context)
            scores_set[0, i_stage, i_window] = decode(X, y_set)

            X1_set1, X2_set2, y1_set1, y2_set2 = split_data_dec(X, y_context, y_splitting=y_set)
            scores_context[1, i_stage, i_window] = decode_xgen(X1_set1, X2_set2, y1_set1, y2_set2, n_jobs=None)

            X1_cxt1, X2_cxt2, y1_cxt1, y2_cxt2 = split_data_dec(X, y_set, y_splitting=y_context)
            scores_set[1, i_stage, i_window] = decode_xgen(X1_cxt1, X2_cxt2, y1_cxt1, y2_cxt2, n_jobs=None)

            scores_shape[0, i_stage, i_window] = decode(X, y_shape)
            scores_width[0, i_stage, i_window] = decode(X, y_width)
            scores_xor[0, i_stage, i_window] = decode(X, y_xor)

            X1_set1, X2_set2, y1_set1, y2_set2 = split_data_dec(X, y_shape, y_splitting=y_set)
            scores_shape[1, i_stage, i_window] = decode_xgen(X1_set1, X2_set2, y1_set1, y2_set2, n_jobs=None)

            X1_set1, X2_set2, y1_set1, y2_set2 = split_data_dec(X, y_width, y_splitting=y_set)
            scores_width[1, i_stage, i_window] = decode_xgen(X1_set1, X2_set2, y1_set1, y2_set2, n_jobs=None)

            X1_set1, X2_set2, y1_set1, y2_set2 = split_data_dec(X, y_xor, y_splitting=y_set)
            scores_xor[1, i_stage, i_window] = decode_xgen(X1_set1, X2_set2, y1_set1, y2_set2, n_jobs=None)

    return scores_context.mean(-1), scores_set.mean(-1), scores_shape.mean(-1), scores_width.mean(-1), scores_xor.mean(
        -1)


def run_decoding_ler_null_shape_locked(data_wins, labels_wins, factors, time_window=[140, 150], n_reps=10):
    dat_stage1 = data_wins[0]
    dat_stage4 = data_wins[3]
    label_stage1 = labels_wins[0]
    label_stage4 = labels_wins[3]

    data_all = np.concatenate((dat_stage1, dat_stage4), axis=2)
    n_cells_stage1 = dat_stage1.shape[2]
    n_cells_stage4 = dat_stage4.shape[2]
    n_cells_all = data_all.shape[2]

    n_stages = len([dat_stage1, dat_stage4])
    scores_context_rnd = np.zeros((n_reps, 2, n_stages))
    scores_set_rnd = np.zeros((n_reps, 2, n_stages))
    scores_shape_rnd = np.zeros((n_reps, 2, n_stages))
    scores_width_rnd = np.zeros((n_reps, 2, n_stages))
    scores_xor_rnd = np.zeros((n_reps, 2, n_stages))

    for rep in range(n_reps):
        print("Rep: ", rep + 1, " of ", n_reps)
        idc_cells = np.arange(n_cells_all)
        random.shuffle(idc_cells)
        dat_stage1_shuffled = data_all[:, :, idc_cells[:n_cells_stage1], :]
        dat_stage4_shuffled = data_all[:, :, idc_cells[n_cells_stage1:], :]
        data_epochs1and4 = [dat_stage1_shuffled, dat_stage4_shuffled]
        labels_epochs1and4 = [label_stage1, label_stage4]
        scores_context_rnd_rep, scores_set_rnd_rep, scores_shape_rnd_rep, scores_width_rnd_rep, scores_xor_rnd_rep = run_decoding_shape_locked(
            data_epochs1and4,
            labels_epochs1and4,
            time_window,
            factors)

        scores_context_rnd[rep, :, :] = scores_context_rnd_rep
        scores_set_rnd[rep, :, :] = scores_set_rnd_rep
        scores_shape_rnd[rep, :, :] = scores_shape_rnd_rep
        scores_width_rnd[rep, :, :] = scores_width_rnd_rep
        scores_xor_rnd[rep, :, :] = scores_xor_rnd_rep

    return scores_context_rnd, scores_set_rnd, scores_shape_rnd, scores_width_rnd, scores_xor_rnd


def plot_dec_xgen(gs, fig, dec_score, dec_score_ler_null, title, y_lim, tails=['smaller', 'smaller'], null_within=None):
    x_data = np.arange(dec_score.shape[1]) + 1
    ax = fig.add_subplot(gs)
    ax.plot(x_data, dec_score[1, :], color='grey')
    ax.scatter(x_data, dec_score[1, :], color='grey')
    ax.plot(x_data, dec_score[0, :], color='black', zorder=5)
    ax.scatter(x_data, dec_score[0, :], color='black', zorder=5)

    ax.set_xticks(x_data)
    ax.set_xticklabels(x_data)
    plt.margins(x=0.1)
    sns.despine(right=True, top=True)
    ax.set_ylabel('accuracy')
    ax.set_xlabel('learning stage')
    ax.set_ylim(y_lim)
    ax.set_title(title)

    print('Cross-generalisation decoding scores')
    p_value_xgen = compute_p_value(dec_score[1, -1], dec_score[1, 0], dec_score_ler_null[:, 1, -1],
                                   dec_score_ler_null[:, 1, 0], tail=tails[1])
    ax.text(2.5, 0.4, p_into_stars(p_value_xgen), fontsize=10, color='grey', ha='center', va='center')
    print('Decoding scores')
    p_value_dec = compute_p_value(dec_score[0, -1], dec_score[0, 0], dec_score_ler_null[:, 0, -1],
                                  dec_score_ler_null[:, 0, 0], tail=tails[0])
    ax.text(2.5, 0.75, p_into_stars(p_value_dec), fontsize=10, color='black', ha='center', va='center')

    if null_within is not None:

        rand_null = null_within.flatten()
        dec_mean = rand_null.mean()
        dec_std = rand_null.std()
        # ax.axhline(dec_mean, linestyle='-', linewidth=1, color=col_chance)
        ax.axhline(dec_mean + 3 * dec_std, linestyle='--', linewidth=1, color='dimgrey', zorder=-7+1)
        ax.axhline(dec_mean - 3 * dec_std, linestyle='--', linewidth=1, color='dimgrey', zorder=-7+1)
        rect = patches.Rectangle((-0.4, dec_mean - 3 * dec_std), 10, 6 * dec_std, edgecolor=None, facecolor='lightgrey',
                                 zorder=-8+1)
        ax.add_patch(rect)

    else:
        ax.axhline(0.5, color='black', linestyle='--', zorder=-5, linewidth=0.8)

    return


def get_sep_tasksets(data_eq, labels_eq):
    idc_half = int(labels_eq[0].shape[-1] / 2)

    data_eq_1 = [data_eq[i_stage][:, :idc_half, :, :] for i_stage in range(len(data_eq))]
    labels_eq_1 = [labels_eq[i_stage][:, :idc_half] for i_stage in range(len(labels_eq))]
    data_eq_2 = [data_eq[i_stage][:, idc_half:, :, :] for i_stage in range(len(data_eq))]
    labels_eq_2 = [labels_eq[i_stage][:, idc_half:] for i_stage in range(len(labels_eq))]

    return data_eq_1, labels_eq_1, data_eq_2, labels_eq_2


def comput_p_val_set_comp(set1_decoding, set2_decoding, set1_decoding_rnd, set2_decoding_rnd, tail='smaller'):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    diff_stage_1 = set2_decoding[0] - set1_decoding[4]
    diff_stage_4 = set2_decoding[-1] - set1_decoding[-1]
    diff_stages = diff_stage_1 - diff_stage_4

    diff_stage_1_rnd = set2_decoding_rnd[:, 0] - set1_decoding_rnd[:, 0]
    diff_stage_4_rnd = set2_decoding_rnd[:, -1] - set1_decoding_rnd[:, -1]
    diff_stages_rnd = diff_stage_1_rnd - diff_stage_4_rnd
    if tail == 'smaller':
        p_val = np.sum(diff_stages_rnd <= diff_stages) / config['ANALYSIS_PARAMS']['N_REPS']
    elif tail == 'greater':
        p_val = np.sum(diff_stages_rnd >= diff_stages) / config['ANALYSIS_PARAMS']['N_REPS']
    elif tail == 'two-sided':
        p_val = np.sum(np.abs(diff_stages_rnd) >= np.abs(diff_stages)) / config['ANALYSIS_PARAMS']['N_REPS']
    return p_val


def run_moving_window_decoding_1var_ler_null(data1, data2, labels1, factor, n_reps=100, time_window=[140, 150],
                                             method='svm', n_jobs=None):
    colour_set1_decoding_rnd = []
    colour_set2_decoding_rnd = []
    for rep in range(n_reps):
        print("Rep: ", rep + 1, " of ", n_reps)

        data_epochs1and4, labels_epochs1and4 = shuffle_stages(data1, data2, labels1)
        data_eq_exp2_1, labels_eq_exp2_1, data_eq_exp2_2, labels_eq_exp2_2 = get_sep_tasksets(data_epochs1and4,
                                                                                              labels_epochs1and4)

        colour_set1_decoding_rnd.append(
            run_moving_window_decoding_1var(data_eq_exp2_1, labels_eq_exp2_1, factor, time_window, method=method,
                                            n_jobs=n_jobs))
        colour_set2_decoding_rnd.append(
            run_moving_window_decoding_1var(data_eq_exp2_2, labels_eq_exp2_2, factor, time_window, method=method,
                                            n_jobs=n_jobs))

    return np.array(colour_set1_decoding_rnd), np.array(colour_set2_decoding_rnd)


def run_moving_window_decoding_1var(data_eq, labels_eq, factor, time_window, method='svm', n_jobs=None):
    n_windows = data_eq[0].shape[0]
    n_stages = len(data_eq)
    decoding = np.zeros((n_stages, n_windows))

    for i_stage in range(n_stages):
        for i_window in range(n_windows):
            X_stage = data_eq[i_stage][i_window, :, :, time_window[0]:time_window[1]].mean(-1)
            y_stage = labels_eq[i_stage][i_window, :]

            y_labels_assigned = assign_lables(y_stage, factor)
            decoding[i_stage, i_window] = decode(X_stage, y_labels_assigned, method=method, n_jobs=n_jobs)

    return decoding.mean(-1)


def run_dec_comparison_tasksets(data1, labels1, data2, labels2, time_window, variables, tails):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    data_eq, labels_eq = split_data_stages_moveavg(data1, labels1, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49,
                                                   if_rnd=True)

    data_eq_exp2_1, labels_eq_exp2_1, data_eq_exp2_2, labels_eq_exp2_2 = get_sep_tasksets(data_eq, labels_eq)

    data_eq_exp1_1, labels_eq_exp1_1 = split_data_stages_moveavg(data2, labels2,
                                                                 n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                                 n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'],
                                                                 trl_min=49, if_rnd=True)

    vars_set1_decoding = []
    vars_set2_decoding = []
    vars_set1_decoding_rnd = []
    vars_set2_decoding_rnd = []
    p_val_vars = []
    for i_var in range(len(variables)):
        var_set1_decoding = run_moving_window_decoding_1var(data_eq_exp1_1 + data_eq_exp2_1,
                                                            labels_eq_exp1_1 + labels_eq_exp2_1,
                                                            factor=variables[i_var],
                                                            time_window=time_window, method='svm', n_jobs=None)

        var_set2_decoding = run_moving_window_decoding_1var(data_eq_exp2_2, labels_eq_exp2_2,
                                                            factor=variables[i_var],
                                                            time_window=time_window, method='svm', n_jobs=None)

        var_set1_decoding_rnd, var_set2_decoding_rnd = run_moving_window_decoding_1var_ler_null(
            data_eq[0], data_eq[-1], labels_eq[0], factor=variables[i_var],
            n_reps=config['ANALYSIS_PARAMS']['N_REPS'],
            time_window=time_window, method='svm', n_jobs=None)

        p_val_var = comput_p_val_set_comp(var_set1_decoding, var_set2_decoding,
                                          var_set1_decoding_rnd, var_set2_decoding_rnd, tail=tails[i_var])

        vars_set1_decoding.append(var_set1_decoding)
        vars_set2_decoding.append(var_set2_decoding)
        vars_set1_decoding_rnd.append(var_set1_decoding_rnd)
        vars_set2_decoding_rnd.append(var_set2_decoding_rnd)
        p_val_vars.append(p_val_var)

    return np.array(vars_set1_decoding), np.array(vars_set2_decoding), np.array(
        [vars_set1_decoding_rnd, vars_set2_decoding_rnd]), p_val_vars


def plot_decoding_set_comp(gs, fig, set1_dec, set2_dec, p_val, y_lim=[0.35, 0.8], title=None):
    ax = fig.add_subplot(gs)
    x_set1 = np.arange(8) + 1
    x_set2 = np.arange(4) + 5
    ax.plot(x_set1, set1_dec, color='grey', zorder=-5)
    ax.scatter(x_set1, set1_dec, color='grey', zorder=-5)

    ax.plot(x_set2, set2_dec, color='black')
    ax.scatter(x_set2, set2_dec, color='black')

    ax.set_xticks(x_set1)
    ax.set_xticklabels(['1', '2', '3', '4', '1', '2', '3', '4'])
    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylim(y_lim)
    ax.set_ylabel('accuracy')
    ax.set_xlabel('exp. 1   exp. 2')
    ax.set_title(title)
    plt.margins(x=0.1)
    sns.despine(top=True, right=True)

    ax.text(6.5, 0.75, p_into_stars(p_val), fontsize=10, color='black', ha='center', va='center')
    return


def run_time_resolved_dec_exp2(data, labels, if_save=True, fname='exp2_dec_time', mode='within_stage', if_exp1=False):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    taskset = np.array(config['ENCODING_EXP2']['taskset'])
    context = np.array(config['ENCODING_EXP2']['context'])
    shape = np.array(config['ENCODING_EXP2']['shape'])
    width = np.array(config['ENCODING_EXP2']['width'])
    xor = np.array(config['ENCODING_EXP2']['xor'])

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)
    data_eq = [data_eq[0], data_eq[-1]]
    labels_eq = [labels_eq[0], labels_eq[-1]]

    if (mode == 'only_ler_null') & (if_exp1 is False):
        dec_rnd_time_all = []
        for i_rep in range(config['ANALYSIS_PARAMS']['N_REPS']):
            print("Rep: ", i_rep + 1, " of ", config['ANALYSIS_PARAMS']['N_REPS'])
            data_epochs1and4, labels_epochs1and4 = shuffle_stages(data_eq[0], data_eq[-1], labels_eq[0])
            dec_rnd_time = []
            for i_time in range(30, 200):
                time_sliding = [i_time, i_time + 1]
                print('Time point: ', i_time + 1, ' of ', data_eq[0].shape[-1] - 50)
                dec_rnd_time.append(run_decoding_shape_locked(data_epochs1and4, labels_epochs1and4, time_window=time_sliding,
                                              factors=[context, taskset, shape, width, xor]))
            dec_rnd_time = np.array(dec_rnd_time)
            dec_rnd_time_all.append(dec_rnd_time)
        dec_rnd_time_all = np.array(dec_rnd_time_all)

        if if_save:
            data_exp1 = [dec_rnd_time_all]
            names_exp1 = ['dec_rnd_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_ler_null.pickle')

        return dec_rnd_time_all



    elif (mode == 'within_stage') & (if_exp1 is False):
        dec_time = []
        for i_time in range(30, 200):
            time_sliding = [i_time, i_time + 1]
            print('Time point: ', i_time + 1, ' of ', data_eq[0].shape[-1] - 50)
            dec_time.append(run_decoding_shape_locked(data_eq, labels_eq, time_window=time_sliding,
                                                      factors=[context, taskset, shape, width, xor]))
        dec_time = np.array(dec_time)

        if if_save:
            data_exp1 = [dec_time]
            names_exp1 = ['dec_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_within_stage.pickle')

        return dec_time

    elif (mode == 'within_stage') & (if_exp1 is True):

        context = np.array(config['ENCODING_EXP1']['colour'])
        shape = np.array(config['ENCODING_EXP1']['shape'])
        width = np.array(config['ENCODING_EXP1']['width'])
        xor = np.array(config['ENCODING_EXP1']['xor'])

        dec_time = []
        for i_time in range(30, 200):
            time_sliding = [i_time, i_time + 1]
            print('Time point: ', i_time + 1, ' of ', data_eq[0].shape[-1] - 50)
            dec_time.append(run_decoding_width_locked_exp1(data_eq, labels_eq, time_window=time_sliding,
                                                      factors=[context, shape, width, xor]))
        dec_time = np.array(dec_time)

        if if_save:
            data_exp1 = [dec_time]
            names_exp1 = ['dec_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_within_stage.pickle')

        return dec_time

    elif (mode == 'only_ler_null') & (if_exp1 is True):

        context = np.array(config['ENCODING_EXP1']['colour'])
        shape = np.array(config['ENCODING_EXP1']['shape'])
        width = np.array(config['ENCODING_EXP1']['width'])
        xor = np.array(config['ENCODING_EXP1']['xor'])

        dec_rnd_time_all = []
        for i_rep in range(config['ANALYSIS_PARAMS']['N_REPS']):
            print("Rep: ", i_rep + 1, " of ", config['ANALYSIS_PARAMS']['N_REPS'])
            data_epochs1and4, labels_epochs1and4 = shuffle_stages(data_eq[0], data_eq[-1], labels_eq[0])
            dec_rnd_time = []
            for i_time in range(30, 200):
                time_sliding = [i_time, i_time + 1]
                print('Time point: ', i_time + 1, ' of ', data_eq[0].shape[-1] - 50)
                dec_rnd_time.append(run_decoding_width_locked_exp1(data_epochs1and4, labels_epochs1and4, time_window=time_sliding,
                                               factors=[context, shape, width, xor]))


            dec_rnd_time = np.array(dec_rnd_time)
            dec_rnd_time_all.append(dec_rnd_time)
        dec_rnd_time_all = np.array(dec_rnd_time_all)

        if if_save:
            data_exp1 = [dec_rnd_time_all]
            names_exp1 = ['dec_rnd_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_ler_null.pickle')

        return dec_rnd_time_all


def shattering_dim_exp2(data_win, labels_win, time_window, method='svm', n_jobs=None):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    all_combos = list(itertools.combinations(list(range(16)), int(8)))
    n_combos = int(len(all_combos) / 2)

    decoding_targets = np.zeros((n_combos, 16))
    for i in range(n_combos):
        decoding_targets[i, all_combos[i]] = 1

    scores = np.zeros((n_combos, config['ANALYSIS_PARAMS']['N_WINDOWS']))
    print('Cutting along all possible axes (shattering dimensionality)')
    for combo in tqdm(range(n_combos)):
        for i_win in range(config['ANALYSIS_PARAMS']['N_WINDOWS']):
            X = np.mean(data_win[i_win, :, :, time_window[0]:time_window[1]], axis=-1, keepdims=False)
            y = assign_lables(labels_win[i_win, :], decoding_targets[combo, :])
            scores[combo, i_win] = decode(X, y, method=method, n_jobs=n_jobs, n_inter=1)

    return scores.mean(-1)


def run_time_resolved_sd_exp2(data, labels, if_save=True, fname='exp2_sd_time', mode='within_stage'):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    data_eq, labels_eq = split_data_stages_moveavg(data, labels, n_stages=config['ANALYSIS_PARAMS']['N_STAGES'],
                                                   n_windows=config['ANALYSIS_PARAMS']['N_WINDOWS'], trl_min=49)
    data_eq = [data_eq[0], data_eq[-1]]
    labels_eq = [labels_eq[0], labels_eq[-1]]

    if mode == 'within_stage':
        sd_time = []
        for i_stage in range(2):
            stage_sd = []
            for i_time in range(30, 200):
                time_sliding = [i_time, i_time + 1]
                print('Time point: ', i_time + 1, ' of ', data_eq[0].shape[-1] - 50)
                stage_sd.append(shattering_dim_exp2(data_eq[i_stage], labels_eq[i_stage], time_window=time_sliding))
            stage_sd = np.array(stage_sd)
            sd_time.append(stage_sd)
        sd_time = np.array(sd_time)

        if if_save:
            data_exp1 = [sd_time]
            names_exp1 = ['sd_time']
            save_data(data_exp1, names_exp1, config['PATHS']['output_path'] + fname + '_within_stage.pickle')

        return sd_time


def compute_lr_null_cos_sim(tasks, reps=500):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    data_all = np.concatenate([tasks[0], tasks[-1]], axis=1)
    n_cells_1 = tasks[0].shape[1]
    n_cells_2 = tasks[-1].shape[1]
    n_cells = n_cells_1 + n_cells_2

    cos_sim_rnd = np.zeros((reps, 2, tasks[0].shape[-1]))
    print('Compute learning null for cos. similarity analysis')
    for i_rep in range(reps):
        cell_idc = np.arange(n_cells)
        np.random.shuffle(cell_idc)
        data_stage1_rnd = data_all[:, cell_idc[:n_cells_1], :]
        data_stage4_rnd = data_all[:, cell_idc[n_cells_1:], :]

        for i_var in range(tasks[0].shape[-1]):
            cos_sim_rnd[i_rep, 0, i_var] = cos_sim(data_stage1_rnd[0, :, i_var], data_stage1_rnd[1, :, i_var])
            cos_sim_rnd[i_rep, 1, i_var] = cos_sim(data_stage4_rnd[0, :, i_var], data_stage4_rnd[1, :, i_var])

    return cos_sim_rnd


def plot_cos_sim(gs, fig, obs, ler_rnd, rnd_null, title, y_lim, tail='two'):
    ax = fig.add_subplot(gs)
    x_data = list(range(1, obs.shape[0] + 1))
    ax.plot(x_data, obs, color='black', zorder=5)
    ax.scatter(x_data, obs, color='black', zorder=5)

    ax.set_xticks(x_data)
    ax.set_xticklabels(x_data)
    plt.margins(x=0.1)
    sns.despine(right=True, top=True)
    ax.set_ylabel('Pearson r')
    ax.set_xlabel('learning stage')
    ax.set_ylim(y_lim)
    ax.set_title(title)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    p_value = compute_p_value(obs[-1], obs[0], ler_rnd[:, -1], ler_rnd[:, 0], tail=tail)
    ax.text(2.5, 0.55, p_into_stars(p_value), fontsize=10, color='black', ha='center', va='center')

    rand_null = rnd_null.flatten()
    dec_mean = rand_null.mean()
    dec_std = rand_null.std()
    # ax.axhline(dec_mean, linestyle='-', linewidth=1, color=col_chance)
    # ax.axhline(dec_mean + 2 * dec_std, linestyle='--', linewidth=1, color='grey')
    # ax.axhline(dec_mean - 2 * dec_std, linestyle='--', linewidth=1, color='grey')
    rect = patches.Rectangle((-0.4, dec_mean - 2 * dec_std), 5.5, 4 * dec_std, edgecolor=None, facecolor='lightgrey',
                             zorder=-8)
    ax.add_patch(rect)

def plot_data_comparison(gs, fig, group1, group2, title=None, ylim=None):
    """
    Plots two groups of data using boxplots and scatter plots for direct comparison.

    Parameters:
    - group1 (array-like): Data points for group 1.
    - group2 (array-like): Data points for group 2.
    """
    data = [group1, group2]
    ax = fig.add_subplot(gs)

    # Creating boxplots
    ax.boxplot(data, widths=0.6, positions=[1, 2], patch_artist=True,
               boxprops=dict(facecolor='grey', color='black'),
               medianprops=dict(color='red'),zorder=-5)

    # Adding jittered scatter plots
    jitter1 = 1 + 0.2 * np.random.rand(len(group1)) - 0.1
    jitter2 = 2 + 0.2 * np.random.rand(len(group2)) - 0.1
    ax.scatter(jitter1, group1, color='black')
    ax.scatter(jitter2, group2, color='black')

    # Set plot details
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['set 1', 'set 2'])
    ax.set_ylabel('no reward/reward\nproporiton')
    ax.set_title(title)
    sns.despine(top=True, right=True)
    ax.axhline(1, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylim(ylim)

    from scipy.stats import ttest_ind, permutation_test

    def statistic(x, y):
        return np.mean(x) - np.mean(y)

    #t_stat, p_val = ttest_ind(group2, group1, alternative='greater')

    res = permutation_test((group2, group1), statistic,
                           n_resamples=np.inf, alternative='greater')

    p_val = res.pvalue
    # plot p-value
    ax.text(1.5, 3.5, p_into_stars(p_val), fontsize=12, ha='center')
    print('ATT facilitation effect p-value ', str(round(p_val,3)))

def run_decoding_width_locked_exp1(data_wins, labels_wins, time_window, factors):
    with open('config.yml') as file:
        config = yaml.full_load(file)

    n_stages = len(data_wins)
    scores_context = np.zeros((n_stages, config['ANALYSIS_PARAMS']['N_WINDOWS']))
    scores_shape = np.zeros((n_stages, config['ANALYSIS_PARAMS']['N_WINDOWS']))
    scores_xor = np.zeros((n_stages, config['ANALYSIS_PARAMS']['N_WINDOWS']))
    for i_stage in range(n_stages):
        for i_window in range(config['ANALYSIS_PARAMS']['N_WINDOWS']):
            X = np.mean(data_wins[i_stage][i_window, :, :, time_window[0]:time_window[1]], axis=-1)
            y = labels_wins[i_stage][i_window, :]
            y_context = np.array(assign_lables(y, factors[0]))
            y_shape = np.array(assign_lables(y, factors[1]))
            y_width = np.array(assign_lables(y, factors[2]))
            y_xor = np.array(assign_lables(y, factors[3]))


            X1_set1, X2_set2, y1_set1, y2_set2 = split_data_dec(X, y_context, y_splitting=y_width)
            scores_context[i_stage, i_window] = decode_xgen(X1_set1, X2_set2, y1_set1, y2_set2, n_jobs=None)

            X1_set1, X2_set2, y1_set1, y2_set2 = split_data_dec(X, y_shape, y_splitting=y_width)
            scores_shape[i_stage, i_window] = decode_xgen(X1_set1, X2_set2, y1_set1, y2_set2, n_jobs=None)

            X1_set1, X2_set2, y1_set1, y2_set2 = split_data_dec(X, y_xor, y_splitting=y_width)
            scores_xor[i_stage, i_window] = decode_xgen(X1_set1, X2_set2, y1_set1, y2_set2, n_jobs=None)

    return scores_context.mean(-1), scores_shape.mean(-1), scores_xor.mean(-1)


def correct_labels_for_cross_gen(labels):

    labels_stages_new = []
    for i_stages in range(len(labels)):
        labels_stage = labels[i_stages]
        labels_sessions_new =[]
        for i_sess in range(len(labels_stage)):
            labels_sess = labels_stage[i_sess] - 1
            labels_sessions_new.append(labels_sess)
        labels_stages_new.append(labels_sessions_new)
    return labels_stages_new
