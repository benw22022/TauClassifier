"""
Test
___________________________________________________________
Plot confusion matrix and ROC curve using testing dataset
"""

from scripts.utils import logger
from config.files import testing_files, ntuple_dir
from config.variables import variable_handler
from config.config import config_dict, get_cuts
from scripts.DataGenerator import DataGenerator
from scripts.preprocessing import Reweighter

def test(args):
	"""
	Plots confusion matrix and ROC curve
	:param args: Args parsed by tauclassifier.py
	"""

    # Initialize objects
	reweighter = Reweighter(ntuple_dir, prong=args.prong)
	cuts = get_cuts(args.prong)

	testing_batch_generator = DataGenerator(testing_files, variable_handler, nbatches=50, batch_size=10000, cuts=cuts,
												reweighter=reweighter, prong=args.prong, label="Testing Generator")

	testing_batch_generator.load_model(args.model, config_dict, args.weights)
	_, _, _, baseline_loss, baseline_acc = testing_batch_generator.predict(make_confusion_matrix=True, make_roc=True)

	logger.log(f"Testing Loss = {baseline_loss}		Testing Accuracy = {baseline_acc}")