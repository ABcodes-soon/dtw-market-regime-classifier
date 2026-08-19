"""Shared fixtures: run every test against BOTH the original (src/) and the
fast (src_fast/) pipelines so both implementations are verified."""

import importlib

import pytest

_MODULES = {
    "clustering_engine": ["src.clustering_engine", "src_fast.clustering_engine"],
    "backtester": ["src.backtester", "src_fast.backtester"],
    "data_processing": ["src.data_processing", "src_fast.data_processing"],
}


@pytest.fixture(params=_MODULES["clustering_engine"], ids=["original", "fast"])
def clustering_engine(request):
    """The clustering engine module (original or fast)."""
    return importlib.import_module(request.param)


@pytest.fixture(params=_MODULES["backtester"], ids=["original", "fast"])
def backtester(request):
    """The backtester module (original or fast)."""
    return importlib.import_module(request.param)


@pytest.fixture(params=_MODULES["data_processing"], ids=["original", "fast"])
def data_processing(request):
    """The data-processing module (original or fast)."""
    return importlib.import_module(request.param)
