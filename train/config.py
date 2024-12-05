import argparse
import socket


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='RITE')
parser.add_argument('--num_iterations', type=int, default=5)
parser.add_argument('--criterion', type=str, default='RRLoss')
parser.add_argument('--base_criterion', type=str, default='BCE3Loss')
parser.add_argument('--model', type=str, default='RRWNet')
parser.add_argument('--num_folds', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-04)
parser.add_argument('--num_epochs', type=int, default=None)
parser.add_argument('--base_channels', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--n_proc', type=int, default=1)
parser.add_argument('--data_folder', type=str, default='./_Data/')
parser.add_argument('--version', type=str, default='Journal_paper')
parser.add_argument('--seed', type=int, default=77)
args = parser.parse_args()



### Configuration arguments

num_folds = args.num_folds
active_folds = range(num_folds)

learning_rate = args.learning_rate
num_epochs = args.num_epochs

dataset = args.dataset

model = args.model
in_channels = args.in_channels
out_channels = args.out_channels
if socket.gethostname() == 'hemingway':
    # Reduce the model size for local testing
    args.base_channels = 16
base_channels = args.base_channels
num_iterations = args.num_iterations

criterion = args.criterion
base_criterion = args.base_criterion

n_proc = args.n_proc
gpu_id = args.gpu_id

training_folder = f'__training/{args.version}/{dataset}'

seed = args.seed

if dataset == 'RITE-train':
    images = [
        33, 24, 36, 30, 25, 29, 40, 21, 37, 34, 35, 32, 27, 39, 26, 38, 28, 23,
        31, 22
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': 'RITE/train/av3',
            'pattern': '[0-9]+[.]png'
        },
        'original': {
            'path': 'RITE/train/enhanced',
            'pattern': '[0-9]+[.]png'
        },
        'mask': {
            'path': 'RITE/train/enhanced_masks',
            'pattern': '[0-9]+[.]png'
        }
    }
elif dataset == 'HRF-Karlsson-w1024':
    images = [
        '06_dr', '06_g', '06_h', '07_dr', '07_g', '07_h', '08_dr', '08_g',
        '08_h', '09_dr', '09_g', '09_h', '10_dr', '10_g', '10_h', '11_dr',
        '11_g', '11_h', '12_dr', '12_g', '12_h', '13_dr', '13_g', '13_h',
        '14_dr', '14_g', '14_h', '15_dr', '15_g', '15_h',
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': f'HRF_AVLabel_191219/train_karlsson_w1024/av3',
            'pattern': '[0-9]+_.+[.]png'
        },
        'original': {
            'path': f'HRF_AVLabel_191219/train_karlsson_w1024/enhanced',
            'pattern': '[0-9]+_.+[.]png'
        },
        'mask': {
            'path': f'HRF_AVLabel_191219/train_karlsson_w1024/enhanced_masks',
            'pattern': '[0-9]+_.+[.]png'
        }
    }

else:
    raise ValueError('dataset not supported')

