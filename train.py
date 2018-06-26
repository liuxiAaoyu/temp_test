import argparse, time, os
import logging
from multiprocessing import cpu_count

import mxnet as mx
from mxnet import gluon
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
import numpy as np

from core import data_lodar
from core import network
from core import metrics
from core.loss import MySoftmaxCrossEntropyLoss

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
parser.add_argument('--data-dir', type=str, default='/media/ihorse/Data/tmp/tusimple/train_set',
                    help='training directory of imagenet images, contains train/val subdirs.')
parser.add_argument('--batch-size', type=int, default=1,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-worker', '-j', dest='num_workers', default=6, type=int,
                    help='number of workers of dataloader.')
parser.add_argument('--gpus', type=str, default='0',
                    help='ordinates of gpus to use, can be "0,1,2" or empty for cpu only.')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123.')
parser.add_argument('--mode', type=str, default='hybrid',
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', default='densenet121', type=str,
                    help='path to checkpoint prefix, default is current working dir')
parser.add_argument('--use_thumbnail', action='store_true',
                    help='use thumbnail or not in resnet. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--use-pretrained', action='store_false',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--prefix', default='models', type=str,
                    help='path to checkpoint prefix, default is current working dir')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='starting epoch, 0 for fresh training, > 0 to resume')
parser.add_argument('--resume', type=str, default='/home/ihorse/Documents/temp_test/densenet121HDCUDC.params',
                    help='path to saved weight where you want resume')
parser.add_argument('--lr-factor', default=0.8, type=float,
                    help='learning rate decay ratio')
parser.add_argument('--lr-steps', default='5,10,15,20,25,30,25,40,45', type=str,
                    help='list of learning rate decay epochs as in str')
parser.add_argument('--dtype', default='float32', type=str,
                    help='data type, float32 or float16 if applicable')
parser.add_argument('--save-frequency', default=1, type=int,
                    help='epoch frequence to save model, best model will always be saved')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer/module.')
parser.add_argument('--log-interval', type=int, default=10,
                    help='Number of batches to wait before logging.')
parser.add_argument('--profile', action='store_false',
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
lr_steps = [int(x) for x in opt.lr_steps.split(',') if x.strip()]
metric = CompositeEvalMetric([metrics.AccWithIgnoreMetric(ignore_label=5), metrics.IoUMetric(ignore_label=5, label_num=2), metrics.SoftmaxLoss(ignore_label=255, label_num=2) ])

net = network.DenseNet_x(classes=2)
net.load_params(opt.resume)#, ctx=context)

CPU_COUNT = cpu_count()
image_list = data_lodar.image_list(opt.data_dir, 0.9)
train_dataset = data_lodar.TuSimpleDataset(opt.data_dir, image_list.train_list, data_lodar.joint_transform)
valid_dataset = data_lodar.TuSimpleDataset(opt.data_dir, image_list.valid_list, data_lodar.joint_transform_valid)
train_data_loader = gluon.data.DataLoader(train_dataset, batch_size, shuffle=True, last_batch='keep', num_workers=CPU_COUNT)
valid_data_loader = gluon.data.DataLoader(valid_dataset, batch_size=1, last_batch='keep', num_workers=CPU_COUNT)


def test(ctx, val_data):
    metric.reset()
    for batch_idx, (data,label) in enumerate(val_data):
        # outputs=[]
        # for x in data:
        #     outputs.append(net(x))
        # metric.update(label, outputs)
        data = data.as_in_context(ctx[0])
        label = label.as_in_context(ctx[0])
        outputs = net(data)
        metric.update([label], [outputs])
    return metric.get()

def update_learning_rate(lr, trainer, epoch, ratio, steps):
    new_lr = lr*(ratio**int(np.sum(np.array(steps)<epoch)))
    trainer.set_learning_rate(new_lr)
    return trainer

def save_checkpoint(epoch, top1, best_acc):
    if opt.save_frequency and (epoch + 1) % opt.save_frequency == 0:
        fname = os.path.join(opt.prefix, '%s_%d_acc_%.4f.params' % (opt.model, epoch, top1))
        net.save_params(fname)
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, top1)
    if top1 > best_acc[0]:
        best_acc[0] = top1
        fname = os.path.join(opt.prefix, '%s_best.params' % (opt.model))
        net.save_params(fname)
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, top1)

def train(opt,ctx):
    if isinstance(ctx, mx.Context):
        ctx=[ctx]
    kv = mx.kv.create(opt.kvstore)
    net.collect_params().reset_ctx(ctx)
    #trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum':opt.momentum, 'multi_precision': True}, kvstore=kv)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': opt.lr}, kvstore=kv)
    
    #loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    loss = MySoftmaxCrossEntropyLoss(axis=1)
    total_time = 0
    num_epochs = 0
    best_acc = [0]
    for epoch in range(opt.start_epoch, opt.epochs):
        trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor,lr_steps)
        tic = time.time()
        metric.reset()
        btic = time.time()
        for i, (data, label) in enumerate(train_data_loader):

            data = data.as_in_context(ctx[0])
            label = label.as_in_context(ctx[0])
            
            with mx.autograd.record():
                output = net(data)
                L = loss(output, label)  
                L.backward()
            trainer.step(data.shape[0])
            metric.update([label],[output])
            if opt.log_interval and not (i+1)%opt.log_interval:
                name, acc = ['0','0','0'],['0','0','0']
                name, acc = metric.get()#['0','0'],['0','0']#
                logger.info('Epoch[%d] batch [%d]\t Speed: %f samples/sec\t %s: %s %s:%s %s:%s %f'%(
                    epoch, i, batch_size/(time.time()-btic),name[0],acc[0],name[1],acc[1],name[2],acc[2],L.mean().asnumpy()
                ))
            btic = time.time()
        epoch_time = time.time()-tic
        
        if num_epochs>0:
            total_time = total_time + epoch_time
        num_epochs = num_epochs + 1

        name, acc = metric.get()
        logger.info('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name[0], acc[0], name[1], acc[1]))
        logger.info('[Epoch %d] time cost: %f'%(epoch, epoch_time))
        name, val_acc = test(ctx, valid_data_loader)
        logger.info('[Epoch %d] validation: %s=%f, %s=%f'%(epoch, name[0], val_acc[0], name[1], val_acc[1]))
        #val_acc=[0,0]
        # save model if meet requirements
        save_checkpoint(epoch, val_acc[0], best_acc)
    if num_epochs > 1:
        print('Average epoch time: {}'.format(float(total_time)/(num_epochs - 1)))



def main():
    net.hybridize()
    train(opt,context)

if __name__ == '__main__':
    if opt.profile:
        # import hotshot, hotshot.stats
        # prof = hotshot.Profile('image-classifier-%s-%s.prof'%(opt.model, opt.mode))
        # prof.runcall(main)
        # prof.close()
        # stats = hotshot.stats.load('image-classifier-%s-%s.prof'%(opt.model, opt.mode))
        # stats.strip_dirs()
        # stats.sort_stats('cumtime', 'calls')
        # stats.print_stats()
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
        prof.runcall(main)
        prof.disable()
        prof.dump_stats('train.prof')

    else:
        main()
