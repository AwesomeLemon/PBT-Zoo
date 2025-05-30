import importlib

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter


class ConfigSpaceSearchSpace:
    def __init__(self, hyperparameters, seed):
        cs_dict = {}
        module = importlib.import_module("ConfigSpace")
        for hp in hyperparameters:
            hp_class = getattr(module, hp.type)
            hp_kwargs = {k: v for k, v in hp.items() if k not in ["type", "distribution", "distribution_kwargs", "name"]}

            if 'distribution' in hp:
                distr_class = getattr(module, hp.distribution)
                distr_kwargs = hp.get("distribution_kwargs", {})
                hp_kwargs['distribution'] = distr_class(**distr_kwargs)

            cs_dict[hp.name] = hp_class(hp.name, **hp_kwargs)
        self.cs = ConfigurationSpace(space=cs_dict, seed=seed)
        print(f'{self.cs}')
        self.n_vars = len(hyperparameters)

    def sample_dict(self, task=None, is_init_sample=False):
        sampled_dict = dict(self.cs.sample_configuration())
        if is_init_sample and hasattr(task, 'init_sample'):
            sampled_dict = task.init_sample(sampled_dict)
        return sampled_dict

    def sample(self, task=None, is_init_sample=False):
        sampled_dict = self.sample_dict(task, is_init_sample)
        sampled = list(sampled_dict.values())
        return sampled

    def get_hp(self, idx):
        return self.cs[self.cs.get_hyperparameter_by_idx(idx)]

    def get_hp_name(self, idx):
        return self.cs.get_hyperparameter_by_idx(idx)

    def get_hp_by_name(self, name):
        return self.cs[name]

    def get_hp_bounds_numerical(self, idx):
        hp = self.get_hp(idx)
        assert isinstance(hp, UniformFloatHyperparameter) or isinstance(hp, UniformIntegerHyperparameter)
        return hp.lower, hp.upper

    def get_hp_names(self):
        return list(self.cs.keys())

    def get_bounds_cont(self, treat_int_as_cont=False):
        bounds = {}
        for i in range(self.n_vars):
            hp_name = self.cs.get_hyperparameter_by_idx(i)
            hp = self.get_hp_by_name(hp_name)
            if isinstance(hp, UniformFloatHyperparameter):
                bounds[hp_name] = self.get_hp_bounds_numerical(i)
            else:
                if treat_int_as_cont and isinstance(hp, UniformIntegerHyperparameter):
                    bounds[hp_name] = self.get_hp_bounds_numerical(i)

        return bounds

    def get_bounds_noncont(self, treat_int_as_cont=False):
        bounds = {}
        for i in range(self.n_vars):
            hp_name = self.cs.get_hyperparameter_by_idx(i)
            hp = self.get_hp_by_name(hp_name)
            if isinstance(hp, CategoricalHyperparameter):
                bounds[hp_name] = 0, len(hp.choices) - 1
            elif isinstance(hp, OrdinalHyperparameter):
                raise NotImplementedError()
            else:
                if (not treat_int_as_cont) and isinstance(hp, UniformIntegerHyperparameter):
                    bounds[hp_name] = self.get_hp_bounds_numerical(i)
        return bounds

    def get_idx_by_value(self, name, value):
        hp = self.get_hp_by_name(name)
        if isinstance(hp, CategoricalHyperparameter):
            return hp.choices.index(value)
        elif isinstance(hp, UniformIntegerHyperparameter):
            return value - hp.lower
        else:
            raise ValueError(name, value)

    def get_choices_by_name(self, name):
        hp = self.get_hp_by_name(name)
        if isinstance(hp, CategoricalHyperparameter):
            return hp.choices
        else:
            if isinstance(hp, UniformIntegerHyperparameter):
                return list(range(hp.lower, hp.upper + 1))
            raise ValueError(name)


    def get_fns_to_convert_from_encoding(self, treat_int_as_cont=False):
        '''
        PB2(-Mix) encodes categorical hyperparameters as indices of the choices => need to convert back
        '''
        fns = []
        for i in range(self.n_vars):
            hp = self.get_hp(i)
            fns.append(self.get_fn_to_convert_from_encoding(hp, treat_int_as_cont))
        return fns

    def get_fn_to_convert_from_encoding(self, hp, treat_int_as_cont=False):
        if isinstance(hp, UniformFloatHyperparameter):
            return float
        elif isinstance(hp, CategoricalHyperparameter):
            choices = hp.choices
            return lambda x_idx: choices[round(x_idx)]
        elif isinstance(hp, UniformIntegerHyperparameter):
            lower = hp.lower
            upper = hp.upper
            if treat_int_as_cont:
                # clip before rounding
                return lambda x: round(max(lower, min(upper, x)))
            else:
                return lambda x_idx: lower + round(x_idx)

        raise NotImplementedError(type(hp))

    def get_cont_cat_ord_types(self, treat_int_as_cont=False):
        types = []
        for i in range(self.n_vars):
            hp = self.get_hp(i)
            if isinstance(hp, UniformFloatHyperparameter):
                types.append('cont')
            elif isinstance(hp, CategoricalHyperparameter):
                types.append('cat')
            elif isinstance(hp, OrdinalHyperparameter):
                types.append('ord')
            else:
                if isinstance(hp, UniformIntegerHyperparameter):
                    if treat_int_as_cont:
                        types.append('cont')
                    else:
                        types.append('cat') # I believe "ordinal" is not useful to me. BG-PBT uses uniforminteger as ordinal, all the other algos treat them as categorical
                else:
                    raise NotImplementedError(type(hp))
        return types

    def vector_to_dict(self, vector):
        hp_names = self.get_hp_names()
        config_dict = {name: vector[hp_names.index(name)] for name in hp_names}
        return config_dict