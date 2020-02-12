# `dm_hard_eight`: DeepMind Hard Eight Task Suite

*DeepMind Hard Eight Tasks* is a set of 8 diverse machine-learning tasks that
require exploration in partially observable environments to solve.

![Hard Eight video](docs/dm_hard_eight.gif)

## Overview

These tasks are provided through pre-packaged
[Docker containers](http://www.docker.com).

This package consists of support code to run these Docker containers. You
interact with the task environment via a
[`dm_env`](http://www.github.com/deepmind/dm_env) Python interface.

Please see the [documentation](docs/index.md) for more detailed information on
the available tasks, actions and observations.

## Requirements

The Hard Eight tasks are intended to be run on Linux and are not officially
supported on Mac and Windows. However, they can in principle be run on any
platform. In particular, on Windows, you may need to run the Python code from
within [WSL](https://docs.microsoft.com/en-us/windows/wsl/about).

`dm_hard_eight` requires [Docker](https://www.docker.com),
[Python](https://www.python.org/) 3.6.1 or later and a x86-64 CPU with SSE4.2
support. We do not attempt to maintain a working version for Python 2.

Note: We recommend using
[Python virtual environment](https://docs.python.org/3/tutorial/venv.html) to
mitigate conflicts with your system's Python environment.

Download and install Docker:

*   For Linux, install [Docker-CE](https://docs.docker.com/install/)
*   Install Docker Desktop for
    [OSX](https://docs.docker.com/docker-for-mac/install/) or
    [Windows](https://docs.docker.com/docker-for-windows/install/).

## Installation

You can install `dm_hard_eight` by cloning a local copy of our GitHub
repository:

```bash
$ git clone https://github.com/deepmind/dm_hard_eight.git
$ pip install ./dm_hard_eight
```

To also install the dependencies for the `examples/`, install with:

```bash
$ pip install ./dm_hard_eight[examples]
```

## Usage

Once `dm_hard_eight` is installed, to instantiate a `dm_env` instance run the
following:

```python
import dm_hard_eight

settings = dm_hard_eight.EnvironmentSettings(seed=123,
    level_name='ball_room_navigation_cubes')
env = dm_hard_eight.load_from_docker(settings)
```

## Citing

If you use `dm_hard_eight` in your work, please cite the accompanying paper:

```bibtex
@article{paine2019making,
  title={Making Efficient Use of Demonstrations to Solve Hard Exploration Problems},
  author={Tom Le Paine and
          Caglar Gulcehre and
          Bobak Shahriari and
          Misha Denil and
          Matt Hoffman and
          Hubert Soyer and
          Richard Tanburn and
          Steven Kapturowski and
          Neil Rabinowitz and
          Duncan Williams and
          Gabriel Barth-Maron and
          Ziyu Wang and
          Nando de Freitas and
          Worlds Team}
  journal={arXiv preprint arXiv:1909.01387},
  year={2019}
}
```

## Notice

This is not an officially supported Google product.
