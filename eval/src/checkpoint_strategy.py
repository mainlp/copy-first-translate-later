from abc import ABC, abstractmethod


class CheckpointStrategy(ABC):
    """
    Interface for a checkpoint evaluation strategy.
    Subclasses define how to list checkpoints and how to load/restore a model.
    """

    @abstractmethod
    def build_initial(self):
        """
        Return tokenizer and config before any checkpoint is loaded.
        Returns a tuple of (tokenizer, model_config_path_or_obj).
        """
        pass

    @abstractmethod
    def list_checkpoints(self):
        """
        Return a list of checkpoint infos.
        Each checkpoint info is a dict with keys at least:
            - "epoch": int
            - "step": int
            - "path": str  (path to checkpoint directory or files)
        """
        pass

    @abstractmethod
    def load_model_for_checkpoint(self, ckpt_info, device):
        """
        Load or restore and return a model for this checkpoint.
        ckpt_info is one of the dicts returned by list_checkpoints.
        """
        pass

    def step_to_tokens(self, step: int):
        """
        Convert training step to number of tokens seen.
        """
        return step * self.batch_size * self.seq_length
