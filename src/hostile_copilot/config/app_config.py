import argparse
from pathlib import Path
from typing import Any, Generic, TypeVar

from hostile_copilot.config import load_config, OmegaConfig

T = TypeVar("T")

class Bind(Generic[T]):
    def __init__(
        self,
        config_path: str,
        *,
        arg_key: str | None = None,
        action: str | None = None,
    ):
        """
        config_path: Access path in the OmegaConfig
        arg_key: Name of the argparse argument to use for the value (defaults to the property name)
        """
        self.config_path: str = config_path
        self.arg_key: str | None = arg_key
        self.property_name: str | None = None
        self.action: str | None = action

    def __set_name__(self, owner, name: str) -> None:
        self.property_name = name
        # Short hand, assume arg key is the same as property name if not specified
        if self.arg_key is None:
            self.arg_key = name

    def __get__(self, instance, owner) -> T:
        if instance is None:
            return self  # type: ignore[return-value]
        return instance._resolve_bind(self)

    def __set__(self, instance, value) -> None:
        raise AttributeError(
            f"'{self.property_name}' is read-only; "
            f"modify '{self.config_path}' via config.set(...) instead"
        )


class Bindings:
    pass


class DefaultsExtractor:
    def __init__(self, parser: argparse.ArgumentParser):
        self._parser = parser

    def extract(self) -> dict[str, Any]:
        defaults = {}

        for action in self._parser._actions:
            if action.dest != argparse.SUPPRESS and action.default is not argparse.SUPPRESS:
                defaults[action.dest] = action.default
                action.default = argparse.SUPPRESS
        
        # Remove 
        return defaults


class AppConfig(OmegaConfig):
    def __init__(
        self,
        bindings: Bindings,
        args: argparse.Namespace,
        config_path: Path | str | None = None,
        arg_defaults: dict[str, Any] | None = None,
    ):
        if not issubclass(bindings, Bindings):
            raise ValueError("bindings must be a subclass of Bindings")
        
        super().__init__(load_config(config_path, wrapper=None))

        self._args = args
        self._arg_defaults = arg_defaults

        for name, value in vars(bindings).items():
            if isinstance(value, Bind):
                setattr(self.__class__, name, value)

    def _resolve_bind(self, bind: Bind) -> T:
        resolved = False
        r_value = None

        arg_key = bind.arg_key
        if arg_key and hasattr(self._args, arg_key):
            value = getattr(self._args, arg_key)
            if value is not None:
                print(f"Resolved by args [{value}]")
                r_value = value
                resolved = True and bind.action != "append"

        value = self.get(bind.config_path)
        if not resolved and value is not None:
            print(f"Resolved by config [{value}]")
            if r_value is not None and bind.action == "append" and isinstance(r_value, list):
                print("Extending")
                r_value.extend(value)
            else:
                r_value = value
            resolved = True

        if self._arg_defaults:
            value = self._arg_defaults.get(bind.arg_key)
            if not resolved and value is not None:
                print(f"Resolved by arg defaults [{value}]")
                if r_value is not None and bind.action == "append" and isinstance(r_value, list):
                    print("Extending")
                    r_value.extend(value)
                else:
                    r_value = value
                resolved = True

        if not resolved:
            print(f"Resolved by None")
        
        return r_value