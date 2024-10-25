import logging
from glob import glob
import tensorboard_reducer as tbr

logger = logging.getLogger(__name__)

def aggregate_results(training_path):
    tensorboard_path = training_path / 'tensorboard'
    reduce_ops = ("mean", "min", "max", "median", "std", "var")
    events_dict = tbr.load_tb_events(sorted(glob(tensorboard_path.__str__() + '/*')))
    n_scalars = len(events_dict)
    n_steps, n_events = list(events_dict.values())[0].shape
    logger.info("Loaded %s TensorBoard runs with %s scalars and %s steps each", n_events, n_scalars, n_steps)
    reduced_events = tbr.reduce_events(events_dict, reduce_ops)
    output_path = tensorboard_path / "aggregates" / "operation"
    for op in reduce_ops:
        logger.debug("Writing \'%s\' reduction to \'%s-%s\'", op, output_path, op)
    tbr.write_tb_events(reduced_events, output_path.__str__(), overwrite=False)
