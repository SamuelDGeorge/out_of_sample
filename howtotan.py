import tensorflow as tf
import tan.experiments.experiment as tanexp
import tan.model.conditionals as conds
import tan.model.transforms as trans
import tan.experiments.config as econfig


def get_tan_nll(zs, mode='fast'):
    if mode == 'fast':
        conf = {
            'ncomps': 40,
            'base_distribution': 'gaussian',
            'dropout_keeprate_val': None,
            'trans_alpha': None,
            'rescale_init_constant': 1.0,
            'cond_param_irange': 1e-6,
            'first_do_linear_map': True,
            'first_trainable_A': True,
            'standardize': True,
            'trans_state_activation': tf.nn.tanh,
            'trans_funcs': [
                trans.additive_coupling, trans.reverse,
                trans.additive_coupling, trans.reverse,
                trans.additive_coupling, trans.reverse,
                trans.additive_coupling, trans.log_rescale],
            'cond_tied_model': False,
            'cond_func': conds.cond_model,
            'param_nlayers': 2,
        }
    elif mode == 'rnn':
        conf = {
            'ncomps': 40,
            'base_distribution': 'gaussian',
            'dropout_keeprate_val': None,
            'trans_alpha': None,
            'rescale_init_constant': 1.0,
            'cond_param_irange': 1e-6,
            'first_do_linear_map': True,
            'first_trainable_A': True,
            'standardize': True,
            'trans_state_activation': tf.nn.tanh,
            'trans_funcs': [trans.simple_rnn_transform, ],
            'cond_func': conds.rnn_model,
            'param_nlayers': 2,
        }
    else:
        conf = {
            'ncomps': 40,
            'base_distribution': 'gaussian',
            'dropout_keeprate_val': None,
            'trans_alpha': None,
            'rescale_init_constant': 1.0,
            'cond_param_irange': 1e-6,
            'first_do_linear_map': True,
            'first_trainable_A': True,
            'standardize': True,
            'trans_state_activation': tf.nn.tanh,
            'trans_funcs': [
                trans.leaky_transformation,
                 trans.log_rescale, trans.rnn_coupling, trans.reverse,
                 trans.linear_map, trans.leaky_transformation,
                 trans.log_rescale, trans.rnn_coupling, trans.reverse,
                 trans.linear_map, trans.leaky_transformation,
                 trans.log_rescale, trans.rnn_coupling, trans.reverse,
                 trans.linear_map, trans.leaky_transformation,
                 trans.log_rescale, trans.rnn_coupling, trans.reverse,
                 trans.linear_map, trans.leaky_transformation,
                 trans.log_rescale, ],
            'cond_func': conds.rnn_model,
            'param_nlayers': 2,
        }
    config = econfig.RedConfig(**conf)
    tanmodel = tanexp.Experiment(config, None, None,
                                 inputs_pl=zs,
                                 graph=tf.get_default_graph())
    return tanmodel.nll, tanmodel.sampler
