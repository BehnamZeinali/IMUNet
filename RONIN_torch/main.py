"""
This is the modification of the code in https://github.com/Sachini/ronin/blob/master/source/ronin_resnet.py
More datasets and machine learning models have been added to the code

Herath, S., Yan, H. and Furukawa, Y., 2020, May. RoNIN: Robust Neural Inertial Navigation in the Wild:
Benchmark, Evaluations, & New Methods. In 2020 IEEE International Conference on Robotics and Automation
(ICRA) (pp. 3146-3152). IEEE.


"""
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import *
from metric import compute_ate_rte, compute_absolute_trajectory_error
from model_resnet1d import *
from MobileNetV2 import MobileNetV2
from MobileNet import MobileNet
from MnasNet import MnasNet
from IMUNet import *
# from IMUNet_1_Pytorch import IMUNet_1
# from IMUNet_2_Pytorch import IMUNet_2
from EfficientnetB0 import EfficientNetB0
from torch.autograd import Variable


_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}


def get_model(arch):
    n_class = 2
    arch = args.arch
    if args.dataset == 'px4':
        n_class = 3
    if arch == 'ResNet':
        network = ResNet1D(6, n_class, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)

    elif arch == 'MobileNetV2':
        network = MobileNetV2()
    elif arch == 'MobileNet':
        network = MobileNet(channels=[32, 64, 128, 256, 512, 1024], width_multiplier=1)
    elif arch == 'MnasNet':
        network = MnasNet(n_class=2)
        print(network)
    elif arch == 'EfficientNet':
        network = EfficientNetB0(n_class=2)
        print(network)
    elif arch == 'IMUNet':
        network = IMUNet(_input_channel, n_class, DSConv, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
        print(network)
    # elif arch == 'imunet_1':
    # network = IMUNet_1(_input_channel, _output_channel, , [2, 2, 2, 2],
    # base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    # elif arch == 'imunet_2':
    # network = IMUNet_2(2)

    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network


def run_test(network, data_loader, device, eval_mode=True):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ, _, _) in enumerate(data_loader):
        pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all


def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


def get_dataset(root_dir, data_list, args, **kwargs):
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True

    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    elif args.dataset == 'ridi':
        seq_type = RIDIGlobSpeedSequence
    elif args.dataset == 'px4':
        seq_type = PX4Sequence
    elif args.dataset == 'oxiod':
        seq_type = OXIODSequence
    elif args.dataset == 'proposed':
        seq_type = ProposedSequence

    dataset = StridedSequenceDataset(
        seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
        random_shift=random_shift, transform=transforms,
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset


def get_dataset_from_list(root_dir, list_path, args, mode, **kwargs):
    if args.dataset == 'px4':
        if (mode == 'train'):
            root_dir = root_dir + '/training'
            data_list = os.listdir(root_dir)
        else:
            root_dir = root_dir + '/validation'
            data_list = os.listdir(root_dir)
    elif args.dataset == 'oxiod':
        if (mode == 'train'):
            root_dir = root_dir + '/train'
            data_list = os.listdir(root_dir)
        else:
            root_dir = root_dir + '/validation'
            data_list = os.listdir(root_dir)
    else:
        with open(list_path) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']

    return get_dataset(root_dir, data_list, args, **kwargs)


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    print(args.root_dir)
    train_dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    end_t = time.time()
    print('Training set loaded. Feature size: {}, target size: {}. Time usage: {:.3f}s'.format(
        train_dataset.feature_dim, train_dataset.target_dim, end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.root_dir, args.val_list, args, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    summary_writer = None
    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args).to(device)
    print(network)
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    # total_params = network.get_num_params()
    # print('Total number of parameters: ', total_params)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12)
    # torch.autograd.set_detect_anomaly(True)
    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))

    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))
        # summary_writer.add_text('info', 'total_param: {}'.format(total_params))

    step = 0
    best_val_loss = np.inf

    print('Start from epoch {}'.format(start_epoch))
    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.
    init_train_targ, init_train_pred = run_test(network, train_loader, device, eval_mode=False)

    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train')

    if val_loader is not None:
        init_val_targ, init_val_pred = run_test(network, val_loader, device)
        init_val_loss = np.mean((init_val_targ - init_val_pred) ** 2, axis=0)
        val_losses_all.append(np.mean(init_val_loss))
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val')

    try:
        for epoch in range(start_epoch, args.epochs):
            start_t = time.time()
            network.train()
            train_outs, train_targets = [], []
            for batch_id, (feat, targ, _, _) in enumerate(train_loader):
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                pred = network(feat)
                train_outs.append(pred.cpu().detach().numpy())
                train_targets.append(targ.cpu().detach().numpy())
                loss = criterion(pred, targ)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                step += 1
            train_outs = np.concatenate(train_outs, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.average((train_outs - train_targets) ** 2, axis=0)

            end_t = time.time()
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, average loss: {}/{:.6f}'.format(
                epoch, end_t - start_t, train_losses, np.average(train_losses)))
            train_losses_all.append(np.average(train_losses))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train')
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)

            if val_loader is not None:
                network.eval()
                val_outs, val_targets = run_test(network, val_loader, device)
                val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                avg_loss = np.average(val_losses)
                print('Validation loss: {}/{:.6f}'.format(val_losses, avg_loss))
                scheduler.step(avg_loss)
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val')
                val_losses_all.append(avg_loss)
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    if args.out_dir and osp.isdir(args.out_dir):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_best.pt')
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)

            total_epoch = epoch

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training complete')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': total_epoch}, model_path)
        print('Checkpoint saved to ', model_path)

    return train_losses_all, val_losses_all


def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    pos = interp1d(ts_ext, pos, axis=0)(ts)
    return pos


def test_sequence(args):
    if args.dataset != 'px4' and args.dataset != 'oxiod':
        if args.test_path is not None:
            if args.test_path[-1] == '/':
                args.test_path = args.test_path[:-1]
            root_dir = osp.split(args.test_path)[0]
            test_data_list = [osp.split(args.test_path)[1]]
        elif args.test_list is not None:
            root_dir = args.root_dir
            with open(args.test_list) as f:
                test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
        else:
            raise ValueError('Either test_path or test_list must be specified.')
    else:
        root_dir = args.root_dir + '/validation'
        test_data_list = os.listdir(root_dir)
        if args.dataset == 'px4':
            args.step_size = 1
    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # if not torch.cuda.is_available() or args.cpu:
    device = torch.device('cpu')
    checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    # else:
    # device = torch.device('cuda:0')
    # checkpoint = torch.load(args.model_path)

    # Load the first sequence to update the input and output size
    _ = get_dataset(root_dir, [test_data_list[0]], args)

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args)
    print(network)
    network.load_state_dict(checkpoint['model_state_dict'])
    # print(network)

    if args.dataset == 'proposed':
        from torchsummary import summary
        dummy_input = Variable(torch.randn(1, 6, 200))
        torch.onnx.export(network, dummy_input, args.out_dir + '/model.onnx')
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []

    pred_per_min = 200 * 60

    for data in test_data_list:
        seq_dataset = get_dataset(root_dir, [data], args, mode='test')
        seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
        ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=np.int)

        targets, preds = run_test(network, seq_loader, device, True)
        losses = np.mean((targets - preds) ** 2, axis=0)
        preds_seq.append(preds)
        targets_seq.append(targets)
        losses_seq.append(losses)

        if args.dataset != 'px4':
            pos_pred = recon_traj_with_preds(seq_dataset, preds)[:, :2]
            pos_gt = seq_dataset.gt_pos[0][:, :2]

            traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
            ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
            ate_all.append(ate)
            rte_all.append(rte)
            pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

            print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))

            # Plot figures
            kp = preds.shape[1]
            if kp == 2:
                targ_names = ['vx', 'vy']
            elif kp == 3:
                targ_names = ['vx', 'vy', 'vz']

            plt.figure('{}'.format(data), figsize=(16, 9))
            plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
            plt.plot(pos_pred[:, 0], pos_pred[:, 1])
            plt.plot(pos_gt[:, 0], pos_gt[:, 1])
            plt.title(data)
            plt.axis('equal')
            plt.legend(['Predicted', 'Ground truth'])
            plt.subplot2grid((kp, 2), (kp - 1, 0))
            plt.plot(pos_cum_error)
            plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
            for i in range(kp):
                plt.subplot2grid((kp, 2), (i, 1))
                plt.plot(ind, preds[:, i])
                plt.plot(ind, targets[:, i])
                plt.legend(['Predicted', 'Ground truth'])
                plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
            plt.tight_layout()

            if args.show_plot:
                plt.show()

            if args.out_dir is not None and osp.isdir(args.out_dir):
                np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                        np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
                plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

            plt.close('all')

        else:

            pos_pred = np.cumsum(preds, axis=0)
            # pos_gt = seq_dataset.gt_pos[0][args.window_size:, ] 
            pos_gt = np.cumsum(targets, axis=0)

            ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
            ate_all.append(ate)
            pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

            print('Sequence {}, loss {} / {}, ate {:.6f}'.format(data, losses, np.mean(losses), ate))

            # Plot figures
            kp = preds.shape[1]
            if kp == 2:
                targ_names = ['vx', 'vy']
            elif kp == 3:
                targ_names = ['vx', 'vy', 'vz']

            plt.figure('{}'.format(data), figsize=(16, 9))

            plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1, projection='3d')
            plt.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2])
            plt.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2])
            plt.title(data)
            # plt.axis('equal')
            plt.legend(['Predicted', 'Ground truth'])
            plt.subplot2grid((kp, 2), (kp - 1, 0))
            plt.plot(pos_cum_error)
            plt.legend(['ATE:{:.3f}'.format(ate_all[-1])])
            for i in range(kp):
                plt.subplot2grid((kp, 2), (i, 1))
                plt.plot(ind, preds[:, i])
                plt.plot(ind, targets[:, i])
                plt.legend(['Predicted', 'Ground truth'])
                plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
            plt.tight_layout()

            if args.show_plot:
                plt.show()

            if args.out_dir is not None and osp.isdir(args.out_dir):
                np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                        np.concatenate([pos_pred, pos_gt], axis=1))
                plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

            plt.close('all')
    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)
    if args.dataset != 'px4':
        # Export a csv file
        if args.out_dir is not None and osp.isdir(args.out_dir):
            with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
                if losses_seq.shape[1] == 2:
                    f.write('seq,vx,vy,avg,ate,rte\n')
                else:
                    f.write('seq,vx,vy,vz,avg,ate,rte\n')
                for i in range(losses_seq.shape[0]):
                    f.write('{},'.format(test_data_list[i]))
                    for j in range(losses_seq.shape[1]):
                        f.write('{:.6f},'.format(losses_seq[i][j]))
                    f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))

        print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}'.format(
            np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all)))
    else:
        
        if args.out_dir is not None and osp.isdir(args.out_dir):
            with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
                if losses_seq.shape[1] == 2:
                    f.write('seq,vx,vy,avg,ate\n')
                else:
                    f.write('seq,vx,vy,vz,avg,ate\n')
                for i in range(losses_seq.shape[0]):
                    f.write('{},'.format(test_data_list[i]))
                    for j in range(losses_seq.shape[1]):
                        f.write('{:.6f},'.format(losses_seq[i][j]))
                    f.write('{:.6f},{:6f}\n'.format(losses_avg[i], ate_all[i]))

        print('----------\nOverall loss: {}/{}, avg ATE:{}'.format(
            np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all)))

    return losses_avg


from torch.autograd import Variable


def onnx_convertor():
    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)
    network = get_model(args.arch)

    network.load_state_dict(checkpoint['model_state_dict'])

    print(network)
    from torchsummary import summary
    dummy_input = Variable(torch.randn(1, 6, 30))
    torch.onnx.export(network, dummy_input, '/home/behnam/Desktop/samsungS10data_np_ni/ar_model_no_pose_resnet101.onnx')


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', default='', type=str)
    parser.add_argument('--val_list', type=str, default='')
    parser.add_argument('--test_list', type=str, default='')
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default='', help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='px4',
                        choices=['ronin', 'ridi', 'proposed', 'oxiod', 'px4'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--arch', type=str, default='IMUNet',
             choices=['ResNet', 'MobileNet', 'MobileNetV2','MnasNet', 'EfficientNet', 'IMUNet'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')
    parser.add_argument('--test_status', type=str, default='seen', choices=['seen', 'unseen'])
    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')

    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})
    dataset = args.dataset
    import os

    # Get the current working directory
    current_dir = os.getcwd()
    
    from pathlib import Path
    path = Path(current_dir)
    current_dir = str(path.parent.absolute())

    # Print the current working directory
    print("Current working directory: {0}".format(current_dir))

    if args.mode == 'train':
        if dataset == 'ronin':
            args.train_list = current_dir +'/Datasets/ronin/list_train.txt'
            args.val_list = current_dir + '/Datasets/ronin/list_val.txt'
            args.root_dir = current_dir + '/Datasets/ronin/train_dataset_1'
            args.out_dir = current_dir + '/RONIN_torch/Train_out/' + args.arch + '/ronin'
            
        elif dataset == 'ridi':
            args.train_list = current_dir + '/Datasets/ridi/data_publish_v2/list_train_publish_v2.txt'
            args.val_list = current_dir + '/Datasets/ridi/data_publish_v2/list_test_publish_v2.txt'
            args.root_dir = current_dir + '/Datasets/ridi/data_publish_v2'
            args.out_dir = current_dir + '/RONIN_torch/Train_out/' + args.arch + '/ridi'
        elif dataset == 'proposed':
            args.train_list = current_dir +'/Datasets/proposed/list_train.txt'
            args.val_list = current_dir +'/Datasets/proposed/list_test.txt'
            args.root_dir = current_dir + '/Datasets/proposed'
            args.out_dir = current_dir +'/RONIN_torch/Train_out/' + args.arch + '/proposed'
        elif dataset == 'oxiod':
            args.train_list = ''
            args.val_list = ''
            args.root_dir = current_dir + '/Datasets/oxiod'
            args.out_dir = current_dir +'/RONIN_torch/Train_out/' + args.arch + '/oxiod'
        elif dataset == 'px4':
            args.step_size = 1
            args.train_list = ''
            args.val_list = ''
            args.root_dir = current_dir + '/Datasets/px4'
            args.out_dir = current_dir +'/RONIN_torch/Train_out/' + args.arch + '/px4'
        train(args)
    elif args.mode == 'test':
        if args.test_status == 'unseen':
            if dataset != 'ronin':
                raise ValueError('Undefined mode')
        if dataset == 'ronin':
            args.model_path = current_dir + '/RONIN_torch/Train_out/' + args.arch + \
                              '/ronin/checkpoints/checkpoint_best.pt'
            
            
                            
                              
            
            if args.test_status == 'seen':
                args.root_dir =  current_dir + '/Datasets/ronin/seen_subjects_test_set'
                # args.root_dir = current_dir + '/Datasets/ronin/seen_subjects_test_set'       
                
                args.test_list = current_dir + '/Datasets/ronin/list_test_seen.txt'
                args.out_dir = current_dir + '/RONIN_torch/Test_out/ronin/seen/'  + args.arch
            else:
                args.root_dir = current_dir + '/Datasets/ronin/unseen_subjects_test_set'
                # args.root_dir = current_dir + '/Datasets/ronin/unseen_subjects_test_set'       
                
                args.test_list = current_dir + '/Datasets/ronin/list_test_unseen.txt'
                args.out_dir = current_dir + '/RONIN_torch/Test_out/ronin/unseen/'  + args.arch

        elif dataset == 'ridi':
            args.model_path = current_dir + '/RONIN_torch/Train_out/' + args.arch + \
                              '/ridi/checkpoints/checkpoint_best.pt'
            args.test_list = current_dir + '/Datasets/ridi/data_publish_v2/list_test_publish_v2.txt'
            args.root_dir = current_dir + '/Datasets/ridi/data_publish_v2'
            args.out_dir = current_dir + '/RONIN_torch/Test_out/ridi/' + args.arch
        elif dataset == 'proposed':
            args.model_path = current_dir + '/RONIN_torch/Train_out/' + args.arch + '/proposed/checkpoints' \
                                                                                    '/checkpoint_best.pt'
            args.test_list = current_dir + '/Datasets/proposed/list_test.txt'
            args.root_dir = current_dir + '/Datasets/proposed'
            args.out_dir = current_dir + '/RONIN_torch/Test_out/proposed/' + args.arch
        elif dataset == 'oxiod':
            args.model_path =  current_dir + '/RONIN_torch/Train_out/' + args.arch + '/oxiod/checkpoints/checkpoint_latest.pt'
            args.test_list = current_dir +  '/Datasets/oxiod/'
            args.root_dir = current_dir + '/Datasets/oxiod'
            args.out_dir = current_dir + '/RONIN_torch/Test_out/oxiod/' + args.arch
        elif dataset == 'px4':
            args.model_path = current_dir + '/RONIN_torch/Train_out/' + args.arch + '/px4/checkpoints/checkpoint_latest.pt'
            args.test_list = current_dir +  '/Datasets/px4/'
            args.root_dir = current_dir + '/Datasets/px4'
            args.out_dir = current_dir + '/RONIN_torch/Test_out/px4/' + args.arch
        test_sequence(args)
        # onnx_convertor()
    else:
        raise ValueError('Undefined mode')
