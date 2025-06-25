from odpc.utils.utils import instantiate_from_config
from odpc.evaluation.precision_curse.wandb_reader import WandbReader
from odpc.evaluation.precision_curse.wandb_processor import WandbProcessor
from odpc.evaluation.precision_curse.precision_curse_reporter import PrecisionCurseReporter

class PrecisionCurseAnalyzer:
    def __init__(
            self, 
            name,
            description,
            save_dir,
            reader,
            processor,
            reporter):
        self.name = name
        self.description = description
        self.save_dir = save_dir
        self.reader: WandbReader = instantiate_from_config(reader)
        self.processor: WandbProcessor = instantiate_from_config(processor)
        self.reporter: PrecisionCurseReporter = instantiate_from_config(reporter)


    def run(self):
        wandb_runs = self.reader.read_data(verbose=True)
        processed_data = self.processor.process_data(wandb_runs, verbose=True)
        self.reporter.report(processed_data, verbose=True)

