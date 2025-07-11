from abc import ABC
from neunet import util, constants as const
from typing import Callable


class ConfigError(Exception):
    """
    An Exception subclass for throwing Config-specific errors (like missing fields)
    """

    def __init__(self, msg):
        super().__init__(msg)
        self.message = msg


class Config(ABC):
    """
    Base Config class for all modules
    """

    def __init__(self, config: dict):
        self.conf = config
        self.check_config()
        for k, v in self.conf.items():
            setattr(self, k, v)
        self.typecast()
        if not self.conf.get("pipeline_id"):
            self.pipeline_id = util.get_pipeline_id()
            self.pipeline_id = self.pipeline_id.replace(".", "-")
        # Get logging directory path from constants.py
        self.logging_path = getattr(
            self,
            "logging_path",
            const.LOGGING_DIRECTORY,
        )

    @property
    def required_config(self) -> dict:
        """

        Returns: A dictionary of all required config fields.

            keys: config fields

            values: callable functions to check the corresponding config field value is of proper type

        Implemented in each subclass.

        """
        return dict()

    def check_config(self) -> None:
        """

        Returns: None

        Checks that each field included in the required_config dict is present in the config file; throws a ConfigError
        otherwise.

        Checks that each config value conforms to the type prescribed by the callable values in the required_config
        dict.

        """
        for key in self.required_config.keys():
            if key not in self.conf.keys():
                msg = f"Required config option {key} missing from entered config {self.conf}."
                raise ConfigError(msg)
            if not self.required_config[key](self.conf[key]):
                msg = f"For key {key}, the required config type callable {self.required_config[key]} did not return a Truthy value"
                raise ConfigError(msg)

    def typecast(self) -> None:
        """

        Returns: None

        This method is not used at all yet, but we leave it here so that if there is a config field that needs to be
        forced to a certain type/format we can do it here.

        """
        pass

    def is_a(self, desired_type: type) -> Callable:
        """

        Args:
            desired_type: the type you desire to check

        Returns: a callable function that will check if its input is of the desired type.
        This is for use as a value in the required_config dict.

        """
        return lambda x: isinstance(x, desired_type) or x is None
