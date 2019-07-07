"""
adapted from  Detectron config file (https://github.com/facebookresearch/Detectron)
"""

import os
import yaml
import copy
from ast import literal_eval
from fractions import Fraction
import logging

path = os.path.dirname(__file__)

class AttrDict(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        elif name.startswith('__'):
            raise AttributeError(name)
        else:
            self[name] = AttrDict()
            return self[name]

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

    def __str__(self):
        return yaml.dump(self.strip(), default_flow_style=False)

    def merge(self, other):
        if not isinstance(other, AttrDict):
            other = AttrDict.cast(other)

        for k, v in other.items():
            v = copy.deepcopy(v)
            if k not in self or not isinstance(v, dict):
                self[k] = v
                continue
            AttrDict.__dict__['merge'](self[k], v)

    def strip(self):
        if not isinstance(self, dict):
            if isinstance(self, list) or isinstance(self, tuple):
                self = str(tuple(self))
            return self
        return {k: AttrDict.__dict__['strip'](v) for k, v in self.items()}

    @staticmethod
    def cast(d):
        if not isinstance(d, dict):
            return d
        return AttrDict({k: AttrDict.cast(v) for k, v in d.items()})


def parse(d):
    # parse string as tuple, list or fraction
    if not isinstance(d, dict):
        if isinstance(d, str):
            try:
                d = literal_eval(d)
            except:
                try:
                    d = float(Fraction(d))
                except:
                    pass
        return d
    return AttrDict({k: parse(v) for k, v in d.items()})

def load(fname):
    with open(fname, 'r') as f:
        ret = parse(yaml.load(f))
    return ret


def setup(args, log):
    ldir = os.path.join(path, '../', 'logs')
    if not os.path.exists(ldir):
        os.makedirs(ldir)

    lfile = args.name + '_' + log + '.txt' if log else args.name + '.txt'
    lfile = os.path.join(ldir, lfile)

    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s', filename=lfile)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)

class Parser(AttrDict):
    def __init__(self, cfg_name='', log=''):
        self.add_cfg('PATH')
        if cfg_name:
            self.add_cfg(cfg_name)
            setup(self, log)

    def add_args(self, args):
        self.merge(vars(args))
        return self

    def add_cfg(self, cfg, args=None, update=False):
        if os.path.isfile(cfg):
            fname = cfg
            cfg   = os.path.splitext(os.path.basename(cfg))[0]
        else:
            fname = os.path.join(path, '../experiments', cfg + '.yaml')

        self.merge(load(fname))
        self['name'] = cfg

        if args is not None:
            self.add_args(args)

        if cfg and args and update:
            self.save_cfg(fname)

        return self

    def save_cfg(self, fname):
        with open(fname, 'w') as f:
            yaml.dump(self.strip(), f, default_flow_style=False)

    def getdir(self):
        if 'name' not in self:
            self['name'] = 'testing'

        checkpoint_dir = os.path.join(self.ckpt_dir, self.name)
        return checkpoint_dir

    def makedir(self):
        checkpoint_dir = self.getdir()
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        fname = os.path.join(checkpoint_dir, 'cfg.yaml')
        self.save_cfg(fname)

        return checkpoint_dir
