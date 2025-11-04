import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()


    # model
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--device', type=str, default='auto')
    
    # data
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)

    # augmentation
    parser.add_argument('--default_augment', action='store_true')
    parser.add_argument('--mix', action='store_true',
                        help='whether mixing data with fractal image')
    parser.add_argument('--active_lam', action='store_true',
                        help='sampling alpha value at every iteration')

    # training
    parser.add_argument('--resume', action='store_true',
                        help="Wheter to resuming training with trained_checkpoints")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--warmup_start_lr', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=-1.0) # if <0, deactivate gradient clip

    # log
    parser.add_argument('--use_wandb', action='store_false')
    parser.add_argument('--val_freq', type=int, default=1)

    return parser.parse_args()