import wandb

class WandbLogger():
    # constructor
    def __init__(self, config, **kwargs):
        self._cfg = kwargs.copy()
        self._watch_log_freq = self._cfg.pop("watch_log_freq", 10)

        # wandb.init() automatically looks for an API key in ~/.netrc file or the WANDB_API_KEY environment variable. 
        # If it finds one, you are logged in implicitly.
        self._runner = wandb.init(config=config, **self._cfg)

        self._cfg["config"] = config
        self._cfg["dir"] = self._runner.dir


    # make WandbLogger object to provide the same interface as wandb.Run
    def __getattr__(self, attr):
        return getattr(self._runner, attr)


    """
    def log(self, data, **kwargs):
        self._runner.log(data, **kwargs)


    def finish(self):
        self._runner.finish()
    """

