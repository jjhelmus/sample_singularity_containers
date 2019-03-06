#! /usr/bin/env python

import argparse

from jinja2 import Template


TEMPLATE_NAME = './singularity.template'


def get_recipe_filename(env_name, python_ver):
    """ Return a string for the recipe filename for a given environment. """
    short_pyver = python_ver.replace('.', '')
    return 'Singularity.' + env_name + "_" + short_pyver


def get_pkg_list(env_name, python_ver):
    """ Return a list of packages to add to the environment. """
    # Added to all environments
    pkg_list = [
        'mpi4py',
        'pydotplus',
        'altair',
        'hdbscan',
        'datreant=1.0',
        'pymatgen',
        'tpot',
        'seaborn',
        'nb_conda',
        'libiconv',         # may not be necessary anymore
        'ipyparallel',      # may not be necessary anymore
        'mkl=2019.1=144',   # to ensure mkl is not downgraded
        'libgcc-ng=8',      # to ensure libgcc is not downgraded
        'blas=*=mkl',       # to ensure openblas is not installed
    ]
    if env_name == 'cpu':
        # added to cpu environment
        pkg_list.extend([
            'tensorflow',
            'keras',
            'caffe',
        ])
    else:
        assert env_name.startswith('gpu')
        # added to gpu environments
        pkg_list.extend([
            'tensorflow-gpu',
            'theano',
            'lasagne',
            'keras-gpu',
            'pytorch',
            'torchvision',
            'blaze',
            'libxslt',
            'caffe-gpu',
        ])
        if env_name == 'gpu_cuda80':
            pkg_list.append('cudatoolkit=8.0')
        else:
            assert env_name == 'gpu_cuda90'
            pkg_list.append('cudatoolkit=9.0')
    return pkg_list


def create_recipe(env_name, python_ver, recipe_filename):
    """ Create a singularity recipe for a specific environment

    Parameters
    ----------
    env_name : str
        Name of the environment to create a recipe for.  Must be one of:
            * cpu
            * gpu_cuda80
            * gpu_cuda90
    python_ver : str
        Major.minor Python version, either '2.7' or '3.6'
    recipe_filename : str or None
        Filename to write recipe to.  None will print the recipe.

    """
    recipe_vars = {
        'miniconda_url': ('https://repo.anaconda.com/miniconda/'
                          'Miniconda3-4.5.12-Linux-x86_64.sh'),
        'channels': '-c defaults -c conda-forge -c engility',
        'anaconda_ver': '2018.12',
        'conda_ver': '4.6',
        'python_ver': python_ver,
        'additional_packages': get_pkg_list(env_name, python_ver),
    }
    with open(TEMPLATE_NAME) as f:
        template_text = f.read()
    template = Template(template_text)
    recipe_text = template.render(**recipe_vars)
    if recipe_filename is None:
        print(recipe_text)
    else:
        with open(recipe_filename, 'w') as f:
            f.write(recipe_text)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create Singularity recipes from a template")
    parser.add_argument(
        "--python_ver", nargs='*', action='store', default=['2.7', '3.6'],
        help="major.minor python versions, default is both 2.7 and 3.6")
    parser.add_argument(
        "--env_name", nargs='*', action='store',
        default=['cpu', 'gpu_cuda80', 'gpu_cuda90'],
        help="environment name, cpu, gpu_cuda80 or gpu_cuda90, default is all")
    parser.add_argument(
        "--show", action="store_true",
        help="Print the recipe(s) to stdout, rather than to files")
    return parser.parse_args()


def main():
    args = parse_arguments()
    for env_name in args.env_name:
        for python_ver in args.python_ver:
            if args.show:
                recipe_filename = None
            else:
                recipe_filename = get_recipe_filename(env_name, python_ver)
                print("Writing:", recipe_filename)
            create_recipe(env_name, python_ver, recipe_filename)
    return


if __name__ == "__main__":
    main()
