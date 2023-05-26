import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
import argparse

trackers = []
parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--tracker_param', type=str, help='Name of config file.')
parser.add_argument('--dataset_name', type=str, help='Name of config file.')
parser.add_argument('--run_ids', nargs='+', help='<Required> Set flag', required=False)
args = parser.parse_args()

dataset_name = args.dataset_name

trackers.extend(trackerlist(name='mixformer2_vit_online', parameter_name=args.tracker_param, dataset_name=args.dataset_name,
                            run_ids=args.run_ids, display_name='MixFormerDeit'))

dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
