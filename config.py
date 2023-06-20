import argparse

parser = argparse.ArgumentParser(description='video-content-sentiment-analyse by frames')

# =================== path Configs ===================#
# parser.add_argument('--train_video_dir', type=str, default='/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/train')
# parser.add_argument('--val_video_dir', type=str, default='/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/test')
# parser.add_argument('--model_save_root', type=str, default="./models", help='destination root of saving model')
parser.add_argument('--save_root', type=str, default="./models")

# =================== dataset Configs ===================#
parser.add_argument('--VA', '-VA', type=str, default='VA', choices=['V', 'A', 'VA'])
parser.add_argument('--length', '-l', default=64, type=int, help='frame length')
parser.add_argument('--samplingRate', '-sr', default=1, type=int, help='sampling rate')

# =================== path Configs ===================#
parser.add_argument('--workers', '-j', default=16, type=int, help='number of data loading workers(default: 2)')
parser.add_argument('--video_numbers', '-vn', default=0, type=float, help='ratio of input video.')
parser.add_argument('--batch_size', '-b', default=64, type=int)
parser.add_argument('--cuda_devices', '-c', type=str, default='3')

# =================== optimizer Configs ===================#
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--lr_steps', default=[10, 20], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
# parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--loss_type', '-lt', type=str, default="mse", choices=['mse', 'ccc', 'marco'])
parser.add_argument('--optim', default="RMSprop", type=str)
# =================== Model Configs ===================#
parser.add_argument('--model', '-m', type=str, default='transformer')
parser.add_argument('--modelConfig', '-mc', type=str, default='1')
parser.add_argument('--epochs', default=4, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--early_stop', type=int, default=10)
parser.add_argument('--random_seed', type=int, default=30)

# =================== Monitor Configs ===================#
parser.add_argument('--printFreq', '-p', default=4, type=int, help='print frequency')
parser.add_argument('--eval-freq', '-ef', default=1, type=int, metavar='N', help='evaluation frequency (default: 50) epochs')
parser.add_argument('--logName', '-ln', type=str, default='base')