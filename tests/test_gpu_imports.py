import pytest
import importlib

libs_to_import = [
    'pygpu',
    'torch',   # 'pytorch',
    'torchvision',
    'theano',
    'lasagne',
]


@pytest.mark.parametrize("lib_to_import", libs_to_import)
def test_import(lib_to_import):
    importlib.import_module(lib_to_import)
