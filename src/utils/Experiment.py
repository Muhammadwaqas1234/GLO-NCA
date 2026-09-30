import torch

DEFAULTS = {
    'batch_duplication': 1, 'channel_n': 16, 'cell_fire_rate': 0.5, 'output_channels': 1,
    'input_channels': 1, 'patchify': False, 'priotize_masks': None, 'use_attention': False,
}


class Experiment():
    r"""Experiment: configuration, the data split and the train/val/test state."""
    def __init__(self, config, dataset, model, agent):
        r"""Merge the config with defaults, size the dataset and attach the agent."""
        self.projectConfig = config
        for k, v in DEFAULTS.items():
            self.projectConfig[0].setdefault(k, v)
        self.config = self.projectConfig[0]
        self.dataset = dataset
        self.model = model
        self.agent = agent
        self.data_split = DataSplit()
        self.set_size()
        self.agent.set_exp(self)
        if not self.get_from_config('unlock_CPU'):
            print("Threads are limited to 1 on shared servers; set 'unlock_CPU: True' to lift this.")
            torch.set_num_threads(1)

    def set_size(self):
        r"""Set the dataset's working size; a multi-level input_size uses its last level."""
        first = self.config['input_size'][0]
        if isinstance(first, (tuple, list)):
            self.dataset.set_size(self.config['input_size'][-1])
        else:
            self.dataset.set_size(self.config['input_size'])

    def set_model_state(self, state: str) -> None:
        r"""Switch the dataset split and the models' train/eval mode."""
        self.model_state = state
        self.dataset.set_paths(self.data_split.get_images(state))
        self.dataset.set_state(state)
        models = [self.model] if not isinstance(self.model, list) else self.model
        for m in models:
            if self.model_state == "train":
                m.train()
            else:
                m.eval()

    def get_from_config(self, tag):
        r"""Get a value from the config, or None when absent."""
        return self.config.get(tag)


class DataSplit():
    r"""Train / validation / test case entries, filled by the trainer."""
    def __init__(self):
        r"""Create empty image and label splits."""
        self.images = {'train': {}, 'val': {}, 'test': {}}
        self.labels = {'train': {}, 'val': {}, 'test': {}}

    def get_images(self, state):
        r"""Image entries of one split ('train', 'val' or 'test')."""
        return self.get_data(self.images[state])

    def get_labels(self, state):
        r"""Label entries of one split ('train', 'val' or 'test')."""
        return self.get_data(self.labels[state])

    def get_data(self, data):
        r"""Flatten the {id: {slice: entry}} structure into a list."""
        lst_out = []
        for d in data.values():
            lst_out.extend([*d.values()])
        return lst_out
