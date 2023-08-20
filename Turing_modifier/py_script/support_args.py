import argparse

attrs_default = ['gender', 'age']

def parse(args=None, test=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=0)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='lrelu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=1.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=1.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=1.0)
    parser.add_argument('--update_lambda_rate', dest='update_lambda_rate', type=float, default=0.0)

    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')

    parser.add_argument('--add_noise_to_dfloss', dest='add_noise_to_dfloss', type=str, default='yes')
    parser.add_argument('--add_noise_to_dcloss', dest='add_noise_to_dcloss', type=str, default='yes')

    parser.add_argument('--conf_num', type=int, default=12)
    parser.add_argument('--train_aga_num', type=int, default=3)
    parser.add_argument('--edite_degree', type=float, default=0.0)

    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=32)
    parser.add_argument('--mode', dest='mode', default='dcgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--b_distribution', dest='b_distribution', default='truncated_normal', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--data_n', dest='data_n', type=int, default=0)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='# of epochs')
    parser.add_argument('--iter_num', dest='iter_num', type=int, default=20)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=3)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=10)

    return parser.parse_args(args)
