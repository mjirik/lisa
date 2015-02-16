#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Module for experiment support
"""

import os
import argparse

import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import subprocess
import glob
import numpy as np
import copy

# ----------------- my scripts --------
import misc
import volumetry_evaluation
import io3d
import io3d.misc
# <codecell>


def run_and_make_report(pklz_dirs, labels, markers, sliver_reference_dir,
                        input_data_path_pattern, conf_default=None,
                        conf_list=None, show=True):
    """
    Je potřeba mít připraven adresář s daty(pklz_dirs), v nichž jsou uloženy
    seedy. Dále pak soubory s konfigurací a stejným jménem jako adresář +
    přípona '.conf'.

    :sliver_dir: dir with sliver reference data
    :input_data_path_pattern: Directory containing data with seeds
         "/home/mjirik/exp010-seeds/*-seeds.pklz"
    :pklz_dirs: List of dirs. In each will be output for one experiment. There
         is also required to have file with same name as directory with
         extension.config. In this files is configuration of Lisa.
    """
    sliver_dir = sliver_reference_dir
    yaml_files = [os.path.normpath(path) + '.yaml' for path in pklz_dirs]
    eval_files = [os.path.normpath(path) + '_eval' for path in pklz_dirs]
    exp_conf_files = [os.path.normpath(path) + '.config' for path in pklz_dirs]
    # exp_conf_files = [os.path.join(os.path.normpath(path),
    # 'organ_segmentation.config') for path in pklz_dirs]
    if conf_list is not None:
        logger.info("generate configuration")
        generate_configs(pklz_dirs, conf_default, conf_list)
    logger.info("run experiments")
    print "run experiments"
    # run experiments
    run_all_liver_segmentation_experiments_with_conf(
        exp_conf_files,
        input_data_path_pattern,
        output_paths=pklz_dirs,
        dry_run=False)
    # run evaluation
    print "run evaluation"
    sliver_eval_all_to_yamls(
        yaml_files, pklz_dirs, sliver_dir, eval_files, recalculateThis=None)
    print "make report"
    # make report

    report(pklz_dirs, labels, markers, show=show)


def generate_configs(pklz_dirs, conf_default, conf_list):
    """
    Based on conf_list and conf_default generate configs.

    """
    exp_conf_files = [os.path.normpath(path) + '.config' for path in pklz_dirs]
    # exp_conf_files = [os.path.join(os.path.normpath(path),
    for i in range(0, len(pklz_dirs)):
        conf = copy.copy(conf_default)
        conf.update(conf_list[i])
        io3d.misc.obj_to_file(conf, exp_conf_files[i], filetype='yaml')


def run_all_liver_segmentation_experiments_with_conf(
    exp_conf_files,
    input_data_path_pattern,
    output_paths,
        dry_run=False):
    """
    Only if there is almost empty dir
    """

    for i in range(0, len(exp_conf_files)):

        config_file_path = exp_conf_files[i]
        # head, teil = os.path.split(config_file_path)
        # TODO make more robust
        # look into directory, if there are some file, we expect that this are
        # results

        if os.path.isfile(config_file_path):
            logger.debug('file "%s" found' % (config_file_path))
            if len(glob.glob(output_paths[i] + '/*.pkl*')) < 2:
                logger.debug('performing segmentation experiment')
                run_liver_segmentation_experiment_with_conf(
                    config_file_path,
                    input_data_path_pattern=input_data_path_pattern,
                    output_path=output_paths[i],
                    dry_run=dry_run)
            else:
                logger.info('skipping dir "%s"' % (output_paths[i]))

        else:

            confs = \
                'config file "' + config_file_path +\
                '" does not exist. Create it with "lisa -cf" parameter.'
            logger.warning(confs)

            print confs


def run_liver_segmentation_experiment_with_conf(
        config_file_path=None,
        bsh_lisa="python ./lisa.py -ni ",
        input_data_path_pattern="/home/mjirik/data/medical/processed/spring2014/exp010-seeds/org-liver-orig???.mhd-exp010-seeds.pklz",  # noqa
        output_path=None,
        dry_run=False,
        use_subprocess=False):
    """
    Run experiments with defined config file
    config_file_path:
    dry_run: only print command

    Example::
        run_experiment_with_conf(
            "/home/mjirik/exp026-03bl06sm/organ_segmentation.config",
            dry_run=True)
    """
#     bsh_lisa = "python /home/mjirik/projects/lisa/lisa.py -ni "
# bsh_data = "-dd
# /home/mjirik/data/medical/processed/spring2014/exp010-seeds/org-liver-orig%03i.mhd-exp010-seeds.pklz
# "

    if output_path is None:
        output_path, teil = os.path.split(config_file_path)

    if not os.path.isdir:
        os.mkdir(output_path)
    bsh_output = " -op " + output_path + " "
    filenames = glob.glob(input_data_path_pattern)
    if len(filenames) == 0:
        logger.warning('input_data_path_pattern "%s" is empty'
                       % (input_data_path_pattern))
        print "input data path is empty " + input_data_path_pattern
    for fn in filenames:
        bsh_data = " -dd " + fn + " "

        bsh = bsh_lisa + bsh_data
        if config_file_path is not None:
            bsh_config = "-cf " + config_file_path
            bsh = bsh + bsh_config + bsh_output

        print bsh
        if not dry_run:
            if use_subprocess:
                process = subprocess.Popen(bsh.split(), stdout=subprocess.PIPE)
                output = process.communicate()[0]
                print output
            else:
                import lisa.organ_segmentation
                import sys
                tmpargv = sys.argv
                sys.argv = bsh.split()[1:]
                lisa.organ_segmentation.main()
                sys.argv = tmpargv

# <codecell>


def report(pklz_dirs, labels, markers, show=True, image_basename=''):
    """
    based on
    """
    expn = np.array(range(0, len(markers)))
    expn_labels = labels
    dp_params = {
        'markers': markers,
        'labels': labels,
        'loc': 0,
        'show': show,
        'filename': image_basename
    }
    sp_params = {
        'expn': expn,
        'expn_labels': expn_labels,
        'show': show,
        'filename': image_basename
    }
    yaml_files = [os.path.normpath(path) + '.yaml' for path in pklz_dirs]
    print yaml_files

    eval_files = [os.path.normpath(path) + '_eval' for path in pklz_dirs]
    print eval_files
    data = [misc.obj_from_file(fname + '.pkl', filetype='pkl')
            for fname in eval_files]


    dataplot(data, 'voe', 'Volume Difference Error [%]', **dp_params)
    dataplot(data, 'vd', 'Total Volume Difference [%]', **dp_params)
    dataplot(data, 'processing_time', 'Processing time [s]', **dp_params)
    dataplot(data, 'maxd', 'MaxD [mm]', **dp_params)
    dataplot(data, 'avgd', 'AvgD [mm]', **dp_params)
    dataplot(data, 'rmsd', 'RMSD [mm]', **dp_params)

    print "Souhrn měření"

    vd_mn, tmp = sumplot( data, 'vd', 'Total Volume Difference', **sp_params)
    voe_mn, tmp = sumplot( data, 'voe', 'Volume Difference Error',  **sp_params)
    avgd_mn, tmp = sumplot( data, 'avgd', 'Average Distance',       **sp_params)
    maxd_mn, tmp = sumplot( data, 'maxd', 'Maxiamal Distance',      **sp_params)
    rmsd_mn, tmp = sumplot( data, 'rmsd', 'Square Distance',        **sp_params)

    print "\n"
    print 'vd   ', vd_mn
    print "voe ", voe_mn
    print 'maxd ', maxd_mn
    print 'avgd ', avgd_mn
    print 'rmsd ', rmsd_mn

    print "\n"

    print "Přepočteno na skóre"
    import pandas
    # print tables[0].shape
    # pandas.set_option('display.max_columns', None)
    scoreTotal, scoreMetrics, scoreAll = sliverScoreAll(data)

    tables, indexes, columns = scoreTableEvaluation(scoreMetrics)

    df = pandas.DataFrame(tables[0], index=indexes[0], columns=columns[0])
    print df.to_string()

    dataplot(scoreAll, 'voe', 'Volume Difference Error [points]',
             **dp_params
             )
    dataplot(scoreAll, 'vd', 'Total Volume Difference [points]',
             **dp_params
             )


    dataplot(scoreAll, 'maxd', 'MaxD [mm]', **dp_params)
    dataplot(scoreAll, 'avgd', 'AvgD [mm]', **dp_params)
    dataplot(scoreAll, 'rmsd', 'RMSD [mm]',**dp_params)

    vd_mn, tmp = sumplot(scoreAll, 'vd', 'Total Volume Difference', **sp_params)
    voe_mn, tmp = sumplot(scoreAll, 'voe', 'Volume Difference Error',**sp_params)
    avgd_mn, tmp = sumplot(scoreAll, 'avgd', 'Average Distance', **sp_params)
    maxd_mn, tmp = sumplot(scoreAll, 'maxd', 'Maxiamal Distance',**sp_params)
    rmsd_mn, tmp = sumplot(scoreAll, 'rmsd', 'Square Distance', **sp_params)

    print "Total score"

    scoreTotal, scoreMetrics, scoreAll = sliverScoreAll(data)
    print 'Score total: ', scoreTotal

    plot_total(scoreMetrics, labels=labels, err_scale=0.05, show=show)


def recalculate_suggestion(eval_files):
    """
    Check which evaluation does not exists
    """
    recalculateThis = []
    for i in range(0, len(eval_files)):
        if not os.path.isfile(eval_files[i]):
            recalculateThis.append(i)
    return recalculateThis


def sliver_eval_all_to_yamls(yaml_files, pklz_dirs, sliver_dir, eval_files,
                             recalculateThis=None):
    """
    This is time consuming.
    It can be specified which should be evaluated with recalculateThis=[2,5]
    """
    if recalculateThis is None:
        recalculateThis = recalculate_suggestion(eval_files)

    for i in recalculateThis:
        print "Performing evaluation on: ", pklz_dirs[i]
        a = volumetry_evaluation.evaluateAndWriteToFile(yaml_files[i],
                                                        pklz_dirs[i],
                                                        sliver_dir,
                                                        eval_files[i],
                                                        visualization=False,
                                                        return_dir_lists=True
                                                        )
        print a


def plotone(data, expn, keyword, ind, marker, legend):
    if ind in expn:
        xdata = range(1, len(data[ind][keyword]) + 1)
        plt.plot(
            xdata, data[ind][keyword], marker, label=legend, alpha=0.7, ms=10)


def dataplot(data, keyword, ylabel, expn=None, markers=None, labels=None,
             ymin=None, loc=0, filename='', show=True):
    """
    Plot data. Function is prepared for our dataset (for example 5 measures).

    """
    if expn is None:
        expn = range(0, len(data))
    if markers is None:
        markers = ['kp'] * len(data)

    if labels is None:
        labels = [''] * len(data)

    for i in range(0, len(expn)):
        try:
            marker = markers[i]
            label = labels[i]
        except:
            marker = 'xr'
            label = 'Unknown label'
            pass
        plotone(data, expn, keyword, expn[i], marker, label)
        # plotone(data, expn, keyword, 0, 'ks', '1 gauss')

    # plotone(data, expn, keyword, 1, 'kv', '3 gauss')
    # plotone(data, expn, keyword, 2, 'kp', 'smoothing')
    # plotone(data, expn, keyword, 3, 'k*', 'blowup')

    x1, x2, y1, y2 = plt.axis()
    # necheme grafy od 40 do 50. chceme grafy od nuly
    if ymin is not None:
        if y1 > 0:
            y1 = ymin
    plt.axis((0, x2 + 1, y1, y2))
    plt.ylabel(ylabel)
    print loc
    plt.legend(numpoints=1, loc=loc,
               bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    # plt.savefig('pitomost.png')
    plt.savefig(filename + '-' + keyword + '.pdf', bbox_inches='tight')
    # plt.savefig('-avgd.png')
    if show:
        plt.show()
    plt.close()

# <codecell>


def sumplot(data, keyword, ylabel, expn=None, expn_labels=None, loc=70,
            filename='', labels_rotation=70, ymin=None, show=True):
    """
    Plot data. Function is prepared for our dataset (for example 5 measures).
    expn_labels: Labels for x-axis based aligned to expn
    """
    # print 'new sumplot'
    if expn is None:
        expn = range(0, len(data))

    # plotone(data, expn, keyword, 0, 'ks', '1 gauss')
    # plotone(data, expn, keyword, 1, 'kv', '3 gauss')
    # plotone(data, expn, keyword, 2, 'kp', 'smoothing')
    # plotone(data, expn, keyword, 3, 'k*', 'blowup')

    mn = []
    va = []

    for ind in expn:
        dat = data[ind][keyword]
        mn.append(np.mean(dat))
        va.append(np.var(dat))

    plt.errorbar(expn, mn, fmt='ks', yerr=va)
    x1, x2, y1, y2 = plt.axis()
    # necheme grafy od 40 do 50. chceme grafy od nuly

    if ymin is not None:
        if y1 > 0:
            y1 = ymin
    plt.axis((x1, x2, y1, y2))
    plt.ylabel(ylabel)
    print expn
    plt.xlim([np.min(expn) - 1, np.max(expn) + 1])
    # plt.xticks([1,2,3],['hu','ha', 'te'])

    if expn_labels is not None:
        # expn_labels = np.array(['1 gauss', '3 gauss', 'smoothing', 'blowup'])
        expn_labels = np.array(expn_labels)
        expn_labels = expn_labels[np.array(expn)]

        plt.xticks(expn, expn_labels, rotation=labels_rotation)
    print loc
    # plt.legend(numpoints=1, loc=loc)
    # plt.savefig('pitomost.png')
    plt.savefig(filename + '-' + keyword + '.pdf', bbox_inches='tight')
    # plt.savefig('-avgd.png')
    if show:
        plt.show()
    plt.close()

    return mn, va


def plot_total(scoreMetrics, err_scale=1, expn=None, labels=None,
               labels_rotation=70, ymin=None, filename=None, show=True):
    """
    err_scale: scale factor of error bars, defalut err_scale=1
    ymin: minimum of graph. If it is set to None, it is automatic
    """
    ylabel = 'Overal Score, err bars %.2f of variance' % (err_scale)
    if expn is None:
        expn = range(0, len(scoreMetrics))

    # print scoreMetrics

    mn = np.array([np.mean(oneset.reshape(-1)) for oneset in scoreMetrics])
    va = np.array([np.var(oneset.reshape(-1))
                   for oneset in scoreMetrics]) * err_scale

    plt.errorbar(expn, mn, fmt='ks', yerr=va)
    x1, x2, y1, y2 = plt.axis()
    # necheme grafy od 40 do 50. chceme grafy od nuly

    if ymin is not None:
        if y1 > 0:
            y1 = ymin
    plt.axis((x1, x2, y1, y2))
    plt.ylabel(ylabel)
    print expn
    plt.xlim([np.min(expn) - 1, np.max(expn) + 1])
    # plt.xticks([1,2,3],['hu','ha', 'te'])

    if labels is not None:
        # expn_labels = np.array(['1 gauss', '3 gauss', 'smoothing', 'blowup'])
        labels = np.array(labels)
        labels = labels[np.array(expn)]

        plt.xticks(expn, labels, rotation=labels_rotation)

    if filename is not None:
        plt.savefig(filename + '-total.pdf', bbox_inches='tight')


def sliverScore(measure, metric_type):
    """
    Based on sliver metodics
    http://sliver07.org/p7.pdf

    Slope and intercept comutations:
    https://docs.google.com/spreadsheet/ccc?key=0AkBzbxly5bqfdEJaOWJJUEh5ajVJM05YWGdaX1k5aFE#gid=0

    """
    slope = -1
    intercept = 100

    if metric_type is 'vd':
        slope = -3.90625
    elif metric_type is 'voe':
        slope = -5.31914893617021
    elif metric_type is 'avgd':
        slope = -25
    elif metric_type is 'rmsdd':
        slope = -14, 7058823529412
    elif metric_type is 'maxdd':
        slope = -1, 31578947368421

    score = intercept + np.abs(measure) * slope
    score[score < 0] = 0

    return score


def sliverScoreAll(data):  # , returnScoreEachData=False):
    """
    Computers score by Sliver07
    http://sliver07.org/p7.pdf

    input: dataset = [{'vd':[1.1, ..., 0.1], 'voe':[...], 'avgd':[...],
                      'rmsd':[...], 'maxd':[...]},
                      {'vd':[...], 'voe':[...], ...}
                     ]
    return: scoreTotal, scoreMetrics, scoreAll
        Order of scoreMetrics is [vd, voe, avgd, rmsd, maxd]


    """

    scoreAll = []
    scoreTotal = []
    scoreMetrics = []
    for dat in data:
        score = {
            'vd': sliverScore(dat['vd'], 'vd'),
            'voe': sliverScore(dat['voe'], 'voe'),
            'avgd': sliverScore(dat['avgd'], 'avgd'),
            'rmsd': sliverScore(dat['rmsd'], 'rmsd'),
            'maxd': sliverScore(dat['maxd'], 'maxd'),
        }
        scoreAll.append(score)

        metrics = np.array([
            score['vd'],
            score['voe'],
            score['avgd'],
            score['rmsd'],
            score['maxd']
        ])
        scoreMetrics.append(metrics)

        total = np.mean(np.mean(metrics))
        scoreTotal.append(total)

    # if returnScoreEachData:
    #    return scoreTotal, scoreMetrics, scoreAll, scoreEachData
    return scoreTotal, scoreMetrics, scoreAll


# print scoreMetrics
def scoreTableEvaluation(scoreMetrics):
    tables = []
    indexes = []
    columns = []
    for scoreMetric in scoreMetrics:
        score0 = np.mean(scoreMetric, 0)
        # print scoreMetric.shape
        score0 = score0.reshape([1, scoreMetric.shape[1]])
        # print score0.shape

        scm = np.vstack((scoreMetric, score0))
        #
        score1 = np.mean(scm, 1)
        score1 = score1.reshape([scm.shape[0], 1])
        # print 'sc1', score1
        scm = np.hstack((scm, score1))
        # print 'scm', scm
        tables.append(scm.T)

        # index and column
        index = range(1, scm.shape[1])
        index.append('score')
        # print 'len index', len(index)
        indexes.append(index)

        column = ['VD', 'VOE', 'AvgD', 'RMSD', 'MaxD', 'score']
        # range(1, scm.shape[0])
        # column.append('score')
        # print 'len col', len(column)
        columns.append(column)
        # print scm

    return tables, indexes, columns



def processIt(pklz_dirs, sliver_dir, yaml_files, eval_files, markers, labels):
    import misc
    """
    Funkce vypíše report o celém experimentu.
    """
    data = [misc.obj_from_file(fname + '.pkl', filetype='pkl')
            for fname in eval_files]

    print "Jednotlivá měření"
    dataplot(data, 'voe', 'Volume Difference Error [%]', markers=markers,
             labels=labels, loc=0)
    dataplot(data, 'vd', 'Total Volume Difference [%]', markers=markers,
             labels=labels, loc=0)

    dataplot(data, 'processing_time',
             'Processing time [s]', markers=markers, labels=labels, loc=0)

    dataplot(data, 'maxd', 'MaxD [mm]', markers=markers, labels=labels, loc=0)
    dataplot(data, 'avgd', 'AvgD [mm]', markers=markers, labels=labels, loc=0)
    dataplot(data, 'rmsd', 'RMSD [mm]', markers=markers, labels=labels, loc=0)

    print "Souhrn měření"

    # import "experiment_support.ipynb"

    expn = np.array(range(0, len(labels)))
    expn_labels = labels

    print expn_labels
    print expn

    vd_mn, tmp = sumplot(
        data, 'vd', 'Total Volume Difference', expn, expn_labels)
    voe_mn, tmp = sumplot(
        data, 'voe', 'Volume Difference Error', expn, expn_labels)

    avgd_mn, tmp = sumplot(data, 'avgd', 'Average Distance', expn, expn_labels)
    maxd_mn, tmp = sumplot(
        data, 'maxd', 'Maxiamal Distance', expn, expn_labels)
    rmsd_mn, tmp = sumplot(data, 'rmsd', 'Square Distance', expn, expn_labels)

    print 'vd   ', vd_mn
    print "voe ", voe_mn
    print 'maxd ', maxd_mn
    print 'avgd ', avgd_mn
    print 'rmsd ', rmsd_mn

    print "Přepočteno na skóre"

    import pandas
    # print tables[0].shape
    # pandas.set_option('display.max_columns', None)
    scoreTotal, scoreMetrics, scoreAll = sliverScoreAll(data)

    tables, indexes, columns = scoreTableEvaluation(scoreMetrics)

    df = pandas.DataFrame(tables[0], index=indexes[0], columns=columns[0])
    print df.to_string()
    dataplot(scoreAll, 'voe', 'Volume Difference Error [points]',
             markers=markers, labels=labels, loc=0)
    dataplot(scoreAll, 'vd', 'Total Volume Difference [points]',
             markers=markers, labels=labels, loc=0)

    dataplot(scoreAll, 'maxd',
             'MaxD [mm]', markers=markers, labels=labels, loc=0)
    dataplot(scoreAll, 'avgd',
             'AvgD [mm]', markers=markers, labels=labels, loc=0)
    dataplot(scoreAll, 'rmsd',
             'RMSD [mm]', markers=markers, labels=labels, loc=0)

    vd_mn, tmp = sumplot(
        scoreAll, 'vd', 'Total Volume Difference', expn, expn_labels)
    voe_mn, tmp = sumplot(
        scoreAll, 'voe', 'Volume Difference Error', expn, expn_labels)
    avgd_mn, tmp = sumplot(
        scoreAll, 'avgd', 'Average Distance', expn, expn_labels)
    maxd_mn, tmp = sumplot(
        scoreAll, 'maxd', 'Maxiamal Distance', expn, expn_labels)
    rmsd_mn, tmp = sumplot(
        scoreAll, 'rmsd', 'Square Distance', expn, expn_labels)

    # scoreTotal, scoreMetrics, scoreAll =
    # volumetry_evaluation.sliverScoreAll(data)
    scoreTotal, scoreMetrics, scoreAll = sliverScoreAll(data)
    print 'Score total: ', scoreTotal

    plot_total(scoreMetrics, labels=labels, err_scale=0.05)


def get_subdirs(dirpath, wildcard='*', outputfile='experiment_data.yaml'):

    dirlist = []
    if os.path.exists(dirpath):
        logger.info('dirpath = ' + dirpath)
        # print completedirpath
    else:
        logger.error('Wrong path: ' + dirpath)
        raise Exception('Wrong path : ' + dirpath)

    dirpath = os.path.abspath(dirpath)
    # print 'copmpletedirpath = ', completedirpath
    # import pdb; pdb.set_trace()
    dirlist = {
        o: {'abspath': os.path.abspath(os.path.join(dirpath, o))}
        for o in os.listdir(dirpath) if os.path.isdir(
            os.path.join(dirpath, o))
    }
    # import pdb; pdb.set_trace()

    # print [o for o in os.listdir(dirpath) if
    # os.path.isdir(os.path.abspath(o))]

    #    dirlist.append(infile)
    # print "current file is: " + infile
    misc.obj_to_file(dirlist, 'experiment_data.yaml', 'yaml')
    return dirlist

# Funkce vrati cast 3d dat. Funkce ma tri parametry :
# data - puvodni 3d data
# sp - vektor udavajici zacatek oblasti, kterou chceme nacist napr [10,20,2]
# Poradi : [z,y,x]
# area - (area size) udava velikost oblasti, kterou chceme vratit. Opet
# vektor stejne jako u sp
# Funkce kontroluje prekroceni velikosti obrazku.


def getArea(data, sp, area):
    if((sp[0] + area[0]) > data.shape[0]):
        sp[0] = data.shape[0] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose Z"
    if((sp[1] + area[1]) > data.shape[1]):
        sp[1] = data.shape[1] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose Y"
    if((sp[2] + area[2]) > data.shape[2]):
        sp[2] = data.shape[2] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose X"
    return data[sp[0]:sp[0] + area[0],
                sp[1]:sp[1] + area[1],
                sp[2]:sp[2] + area[2]]


def setArea(data, sp, area, value):

    if((sp[0] + area[0]) > data.shape[0]):
        sp[0] = data.shape[0] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose Z"
    if((sp[1] + area[1]) > data.shape[1]):
        sp[1] = data.shape[1] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose Y"
    if((sp[2] + area[2]) > data.shape[2]):
        sp[2] = data.shape[2] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose X"

    data[sp[0]:sp[0] + area[0], sp[1]:sp[1]
         + area[1], sp[2]:sp[2] + area[2]] = value
    return data

if __name__ == "__main__":
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description='Experiment support')
    parser.add_argument('--get_subdirs', action='store_true',
                        default=None,
                        help='path to data dir')
    parser.add_argument('-o', '--output', default=None,
                        help='output file name')
    parser.add_argument('-i', '--input', default=None,
                        help='input')
    args = parser.parse_args()

    if args.get_subdirs:
        if args.output is None:
            args.output = 'experiment_data.yaml'
        get_subdirs(dirpath=args.input, outputfile=args.output)


#    SectorDisplay2__()
