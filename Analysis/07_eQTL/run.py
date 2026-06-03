#!/usr/bin/env python
"""
Unified eQTL analysis entry point.

Usage:
    python Analysis/07_eQTL/run.py evaluate --model bican [--filter basal_ganglia]
    python Analysis/07_eQTL/run.py evaluate --model borzoi [--filter brain]
    python Analysis/07_eQTL/run.py evaluate --model alphagenome [--filter gtex]
    python Analysis/07_eQTL/run.py plot --organ Basal_ganglia [--bican-filter ...] [--borzoi-filter ...] [--ag-filter ...]
"""
import argparse
import importlib
import os
import sys

# Bootstrap: register the digit-prefixed directory as a proper package
_script_dir = os.path.dirname(os.path.abspath(__file__))
_analysis_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_analysis_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import the package via importlib (can't use normal import for '07_eQTL')
_pkg = importlib.import_module('Analysis.07_eQTL')
_cfg = importlib.import_module('Analysis.07_eQTL.config')

os.chdir(_cfg.PWD)


_REGULAR_ORGANS = ['Brain', 'Basal_ganglia', 'Cortex']
_AGG_ORGANS     = ['Brain_agg', 'Basal_ganglia_agg', 'Cortex_agg']


def cmd_evaluate(args):
    data_mod = importlib.import_module('Analysis.07_eQTL.data')
    eval_mod = importlib.import_module('Analysis.07_eQTL.evaluate')

    if args.json:
        data_mod.set_paths_json(args.json)

    add = args.add_addition
    model_data = data_mod.load_model(args.model, args.filter, add_addition=add)
    suffix = f'_{args.filter}' if args.filter else ''

    output_dir = data_mod.get_output_dir(add_addition=add)

    organ = args.organ
    if organ is None:
        eval_mod.evaluate_model(model_data, organ=None, suffix=suffix, add_addition=add, output_dir=output_dir)
        for o in _AGG_ORGANS:
            eval_mod.evaluate_organ_agg(model_data, organ=o, suffix=suffix, add_addition=add, output_dir=output_dir)
    elif organ in _AGG_ORGANS:
        eval_mod.evaluate_organ_agg(model_data, organ=organ, suffix=suffix, add_addition=add, output_dir=output_dir)
    else:
        eval_mod.evaluate_model(model_data, organ=organ, suffix=suffix, add_addition=add, output_dir=output_dir)


def cmd_plot(args):
    data_mod = importlib.import_module('Analysis.07_eQTL.data')
    plot_mod = importlib.import_module('Analysis.07_eQTL.plot')

    if args.json:
        data_mod.set_paths_json(args.json)

    bican_suffix = f'_{args.bican_filter}' if args.bican_filter else ''
    borzoi_suffix = f'_{args.borzoi_filter}' if args.borzoi_filter else ''
    ag_suffix = f'_{args.ag_filter}' if args.ag_filter else ''

    add = args.add_addition
    output_dir = data_mod.get_output_dir(add_addition=add)

    plot_mod.plot_all(
        organ=args.organ,
        output_dir=output_dir,
        bican_suffix=bican_suffix,
        borzoi_suffix=borzoi_suffix,
        ag_suffix=ag_suffix,
        metric=args.metric,
        ymin=args.ymin,
        ymax=args.ymax,
        plot_num=args.plot_num,
        figx=args.figx,
        figy=args.figy,
    )


def main():
    parser = argparse.ArgumentParser(description='Unified eQTL analysis')
    sub = parser.add_subparsers(dest='command', required=True)

    # evaluate
    p_eval = sub.add_parser('evaluate', help='Run per-tissue AUROC evaluation for a model')
    p_eval.add_argument('--model', required=True,
                        help='Model name as defined in the data-paths JSON (e.g. bican, borzoi, '
                             'alphagenome131k, alphagenome_paper)')
    p_eval.add_argument('--json', type=str, default=None,
                        help='Path to a custom data_paths*.json (default: data_paths.json next to data.py)')
    p_eval.add_argument('--filter', type=str, default=None,
                        help='Track filter (model-specific: bican accepts basal_ganglia/cortex/rna/etc; '
                             'borzoi accepts brain/basal_ganglia/cortex/gtex_brain; '
                             'alphagenome accepts brain/basal_ganglia/cortex/gtex)')
    p_eval.add_argument('--organ', type=str, default=None,
                        choices=['Brain', 'Basal_ganglia', 'Cortex',
                                 'Brain_agg', 'Basal_ganglia_agg', 'Cortex_agg'],
                        help='Evaluate a specific organ; omit to run all 6')
    p_eval.add_argument('--add-addition', action='store_true', default=False,
                        help='Include additional negative variants from .additional.h5 and negative_addition.vcf')

    # plot
    p_plot = sub.add_parser('plot', help='Generate comparison plots from saved CSVs')
    p_plot.add_argument('--json', type=str, default=None,
                        help='Path to a custom data_paths*.json (default: data_paths.json next to data.py)')
    p_plot.add_argument('--organ', type=str, default='Basal_ganglia')
    p_plot.add_argument('--bican-filter', type=str, default=None)
    p_plot.add_argument('--borzoi-filter', type=str, default=None)
    p_plot.add_argument('--ag-filter', type=str, default=None)
    p_plot.add_argument('--metric', type=str, default='AUROC',
                        choices=['AUROC', 'AUPRC', 'AUROC_point', 'AUPRC_point'],
                        help='Metric to plot; *_point variants omit error bars (default: AUROC)')
    p_plot.add_argument('--ymin', type=float, default=None,
                        help='Y-axis lower bound (default: 0)')
    p_plot.add_argument('--ymax', type=float, default=None,
                        help='Y-axis upper bound (default: 1)')
    p_plot.add_argument('--plot-num', action='store_true', default=False,
                        help='Annotate each bar with its AUROC/AUPRC value and each group with n_pos/n_neg counts')
    p_plot.add_argument('--figx', type=float, default=None,
                        help='Figure width in inches')
    p_plot.add_argument('--figy', type=float, default=None,
                        help='Figure height in inches')
    p_plot.add_argument('--add-addition', action='store_true', default=False,
                        help='Read from output.addition/ and save to figure.addition/')

    args = parser.parse_args()
    if args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'plot':
        cmd_plot(args)


if __name__ == '__main__':
    main()
