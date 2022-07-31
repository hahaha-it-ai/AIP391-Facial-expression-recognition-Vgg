from models import vgg, fer_mini
from utils.checkpoint import restore
from utils.logger import Logger

nets = {
    'vgg': vgg.Vgg,
    'fer_mini': fer_mini.Fermini
}


def setup_network(hps):
    net = nets[hps['network']]()

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net
