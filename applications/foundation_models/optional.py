"""Utilities for handling optional dependencies in foundation model ETL."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any, Callable, TypeVar, cast

_F = TypeVar("_F", bound=Callable[..., Any])


def import_modelgenerator():
  """Import and return modelgenerator, raising an informative error if missing."""

  try:
    return importlib.import_module("modelgenerator")
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
      "This functionality requires `modelgenerator`. "
      "Install with `pip install modelgenerator`."
    ) from exc


def require_modelgenerator(func: _F) -> _F:
  """Decorator ensuring modelgenerator is available before calling *func*.

  Use this decorator for AIDOCell and scFoundation-specific functions.

  Examples
  --------
  >>> @require_modelgenerator
  >>> def load_aidocell_model(model_class):
  ...     # Uses modelgenerator
  ...     pass
  """

  @wraps(func)
  def wrapper(*args: Any, **kwargs: Any):
    import_modelgenerator()
    return func(*args, **kwargs)

  return cast(_F, wrapper)


def import_scprint():
  """Import and return scprint, raising an informative error if missing."""

  try:
    return importlib.import_module("scprint")
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
      "This functionality requires `scprint`. "
      "Install with `pip install scprint`."
    ) from exc


def require_scprint(func: _F) -> _F:
  """Decorator ensuring scprint is available before calling *func*.

  Use this decorator for scPRINT-specific functions.

  Examples
  --------
  >>> @require_scprint
  >>> def load_scprint_model(checkpoint_path):
  ...     # Uses scprint
  ...     pass
  """

  @wraps(func)
  def wrapper(*args: Any, **kwargs: Any):
    import_scprint()
    return func(*args, **kwargs)

  return cast(_F, wrapper)


def import_scgpt():
  """Import and return scgpt, raising an informative error if missing."""

  try:
    return importlib.import_module("scgpt")
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
      "This functionality requires `scgpt`. "
      "Install with `pip install scgpt wandb gseapy`. "
      "If you encounter issues with `torchtext` compatibility (common with PyTorch), "
      "use conda/mamba instead: "
      "`mamba install pytorch torchtext==0.18.0 -c pytorch -c conda-forge` "
      "then `pip install scgpt wandb gseapy`."
    ) from exc


def require_scgpt(func: _F) -> _F:
  """Decorator ensuring scgpt is available before calling *func*.

  Use this decorator for scGPT-specific functions.

  Examples
  --------
  >>> @require_scgpt
  >>> def load_scgpt_model(model_dir):
  ...     # Uses scgpt
  ...     pass
  """

  @wraps(func)
  def wrapper(*args: Any, **kwargs: Any):
    import_scgpt()
    return func(*args, **kwargs)

  return cast(_F, wrapper)


def import_bionty():
  """Import and return bionty, raising an informative error if missing."""

  try:
    return importlib.import_module("bionty")
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
      "This functionality requires `bionty`. "
      "Install with `pip install bionty` or `pip install lamindb[bionty]`."
    ) from exc


def require_bionty(func: _F) -> _F:
  """Decorator ensuring bionty is available before calling *func*.

  Use this decorator for scPRINT functions that require bionty/lamin.

  Examples
  --------
  >>> @require_bionty
  >>> def populate_lamin_db():
  ...     # Uses bionty
  ...     pass
  """

  @wraps(func)
  def wrapper(*args: Any, **kwargs: Any):
    import_bionty()
    return func(*args, **kwargs)

  return cast(_F, wrapper)


def import_scdataloader():
  """Import and return scdataloader, raising an informative error if missing."""

  try:
    return importlib.import_module("scdataloader")
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
      "This functionality requires `scdataloader`. "
      "Install with `pip install scdataloader` (typically bundled with scprint)."
    ) from exc


def require_scdataloader(func: _F) -> _F:
  """Decorator ensuring scdataloader is available before calling *func*.

  Use this decorator for scPRINT functions that require scdataloader.

  Examples
  --------
  >>> @require_scdataloader
  >>> def populate_lamin_db():
  ...     # Uses scdataloader.utils.populate_my_ontology
  ...     pass
  """

  @wraps(func)
  def wrapper(*args: Any, **kwargs: Any):
    import_scdataloader()
    return func(*args, **kwargs)

  return cast(_F, wrapper)


def import_torchtext():
  """Import and return torchtext, raising an informative error if missing."""

  try:
    return importlib.import_module("torchtext")
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
      "This functionality requires `torchtext`. "
      "Install with `pip install torchtext==0.18.0`. "
      "If you encounter compatibility issues with PyTorch, "
      "use conda/mamba instead: "
      "`mamba install pytorch torchtext==0.18.0 -c pytorch -c conda-forge`."
    ) from exc


def require_torchtext(func: _F) -> _F:
  """Decorator ensuring torchtext is available before calling *func*.

  Use this decorator for scGPT functions that require torchtext.

  Examples
  --------
  >>> @require_torchtext
  >>> def load_scgpt(model_dir):
  ...     # Uses torchtext.vocab.Vocab
  ...     pass
  """

  @wraps(func)
  def wrapper(*args: Any, **kwargs: Any):
    import_torchtext()
    return func(*args, **kwargs)

  return cast(_F, wrapper)
