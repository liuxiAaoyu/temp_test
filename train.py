import argparse, time, os
import logging
from multiprocessing import cpu_count

import mxnet as mx
from mxnet import gluon
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric

from core import data_lodar
from core import network

logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('train.log')
logger = logging.getLogger()
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logging.debug('\n%s','-' * 100)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset to use. options are mnist, cifar10, imagenet and dummy.')
parser.add_argument('--data-dir', type=str, default='/media/xiaoyu/Document/data/TuSimple_Lane/train_set',
                    help='training directory of imagenet images, contains train/val subdirs.')
parser.add_argument('--batch-size', type=int, default=4,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-worker', '-j', dest='num_workers', default=6, type=int,
                    help='number of workers of dataloader.')
parser.add_argument('--gpus', type=str, default='0',
                    help='ordinates of gpus to use, can be "0,1,2" or empty for cpu only.')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123.')
parser.add_argument('--mode', type=str, default='hybrid',
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')

parser.add_argument('--use_thumbnail', action='store_true',
                    help='use thumbnail or not in resnet. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--prefix', default='models', type=str,
                    help='path to checkpoint prefix, default is current working dir')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='starting epoch, 0 for fresh training, > 0 to resume')
parser.add_argument('--resume', type=str, default='/home/ihorse/Documents/temp_test/densenet121HDCUDC.params',
                    help='path to saved weight where you want resume')
parser.add_argument('--lr-factor', default=0.1, type=float,
                    help='learning rate decay ratio')
parser.add_argument('--lr-steps', default='3,6,9', type=str,
                    help='list of learning rate decay epochs as in str')
parser.add_argument('--dtype', default='float32', type=str,
                    help='data type, float32 or float16 if applicable')
parser.add_argument('--save-frequency', default=10, type=int,
                    help='epoch frequence to save model, best model will always be saved')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer/module.')
parser.add_argument('--log-interval', type=int, default=1,
                    help='Number of batches to wait before logging.')
parser.add_argument('--profile', action='store_true',
                    help='Option to turn on memory profiling for front-end, '\
                         'and prints out the memory usage by python function at the end.')
parser.add_argument('--builtin-profiler', type=int, default=0, help='Enable built-in profiler (0=off, 1=on)')
opt = parser.parse_args()

logger.info('Starting new train task:, %s', opt)
mx.random.seed(opt.seed)
batch_size = opt.batch_size
context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]
num_gpus = len(context)
batch_size *= max(1, num_gpus)
lr_steps = [int(x) for x in opt.ls_steps.split(',') if x.strip()]
metric = CompositeEvalMetric([Accuracy(), TopKAccuracy()])

net = network.DenseNet_x()
net.load_params(opt.resume, ctx=context)

CPU_COUNT = cpu_count()
image_list = data_lodar.image_list(opt.dataset_dir, 0.9)
train_dataset = data_lodar.TuSimpleDataset(opt.dataset_dir, image_list.train_list, data_lodar.joint_transform)
valid_dataset = data_lodar.TuSimpleDataset(opt.dataset_dir, image_list.valid_list, data_lodar.joint_transform_valid)

train_data_loader = gluon.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=CPU_COUNT)
valid_data_loader = gluon.data.DataLoader(valid_dataset, batch_size, num_workers=CPU_COUNT)


def test(ctx, val_data):
    metric.reset()
    for batch_idx, (data,label) in enumerate(val_data):
        outputs=[]
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()

def update_learning_rate(lr, trainer, epoch, ratio, steps):
    new_lr = lr*(ratio**int(np.sum(np.array(steps)<epoch)))
    trainer.set_learning_rate(new_lr)
    return trainer

def save_checkpoint():
    pass

def train(opt,ctx):
    if isinstance(ctx, mx.Context):
        ctx=[ctx]
    kv = mx.kv.create(opt.kvstore)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer()


def main():
    net.hybridize()
    train(opt,context)

if __name__ == '__main__':
    if opt.profile:
        import hotshot, hotshot.stats
        prof = hotshot.Profile('image-classifier-%s-%s.prof'%(opt.model, opt.mode))
        prof.runcall(main)
        prof.close()
        stats = hotshot.stats.load('image-classifier-%s-%s.prof'%(opt.model, opt.mode))
        stats.strip_dirs()
        stats.sort_stats('cumtime', 'calls')
        stats.print_stats()
    else:
        main()