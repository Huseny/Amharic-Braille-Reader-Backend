import hashlib
import json
import ast
import os
from typing import Dict, Any
from collections import OrderedDict


class ExtractDic(OrderedDict):
    '''
    dictionary allowing to access by dic[xxx] as well as dic.xxx syntax, including nested dictionaries:
        m = param.ExtractDic(a = 1, b = {'b1': 2, 'b2':3}, c = 4)
        m.c = 5
        m.d = 6
        print(m.a, m.b.b1, m.b.b2, m.c, m.d)
    Example:
    from params import ExtractDic
    params = ExtractDic(
        data_root = local_config.data_path,
        model_name = 'model/inner_halo_types_m{inner_halo_params.margin}_w{inner_halo_params.loss_weights}',
        fold_test_substrs = ['/cam_7_7/', '/cam_7_8/', '/cam_7_9/'],
        fold_no = 0,
        model_params = ExtractDic(output_channels=3, enc_type='se_resnet50',
                                dec_type='unet_scse',
                                num_filters=16, pretrained=True),
        mean = (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
        std = (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
    )
    ...
    params.save()

    parameters 'data_root' and 'model_name' are required for save() and base_filename() functions.
    parameter 'data_root' is not stored and does not influence on hash
    '''
    def __init__(self, *args, **kwargs):
        super(ExtractDic, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for k,v in self.items():
            assert '.' not in k, "ExtractDic: attribute '" + k + "' is invalid ('.' char is not allowed)"
            if isinstance(v, dict):
                self[k] = ExtractDic(v)
            elif isinstance(v, list):
                self[k] = [ExtractDic(item) if isinstance(item, dict) else item for item in v]

    def __repr__(self):
        def write_item(item, margin='\n'):
            if isinstance(item, dict):
                s = '{'
                margin2 = margin + '    '
                for k, v in item.items():
                    if not k.startswith('__') and k != 'data_root':
                        s += margin2 + "'{0}': ".format(k) + write_item(v, margin=margin2) + ","
                if item.items():
                    s += margin
                s += '}'
            elif isinstance(item, (list, tuple)):
                s = '[' if isinstance(item, list) else '('
                for v in item:
                    if isinstance(v, dict):
                        s += margin + '    '
                    else:
                        s += ' '
                    s += write_item(v, margin=margin + '    ')  + ","
                s += ' ' + (']' if isinstance(item, list) else ')')
            else:
                s = repr(item)
            return s
        return write_item(self)

    def has(self, name):
        '''
        checks if self contains attribute with some name, including recursive, i.e. 'b.b1' etc.
        '''
        names = name.split('.')
        dic = self
        for n in names:
            if not hasattr(dic, n):
                return False
            dic = dic[n]
        return True
                
    def hash(self, shrink_to = 6):
        '''
        hash of dict values, invariant to values order
        '''
        hash_dict = self.copy()
        hash_dict.pop('data_root', None)
        return hashlib.sha1(json.dumps(hash_dict, sort_keys=True).encode()).hexdigest()[:shrink_to]

    def get_model_name(self):
        assert self.has('model_name')
        return self.model_name.format(**self) + '_' + self.hash()

    def get_base_filename(self):
        assert self.has('data_root')
        return os.path.join(self.data_root, self.get_model_name())

    def save(self, file_name=None, verbose = 1, can_overwrite = False, create_dirs = False):
        '''
        save to file adding '.param.txt' to name
        '''
        if file_name is None:
            file_name = os.path.join(self.get_base_filename(), 'param.txt')
        if not can_overwrite:
            assert not os.path.exists(file_name), "Can't save parameters to {}: File exists".format(file_name)
        if create_dirs:
            dir_name = os.path.dirname(os.path.dirname(file_name))
            os.makedirs(dir_name, exist_ok=True)
        dir_name = os.path.dirname(file_name)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_name, 'w+') as f:
            s = str(self)
            s = s + '\nhash: ' + self.hash()
            f.write(s)
            if verbose >= 2:
                print('params: '+ s)
            if verbose >= 1:
                print('saved to ' + str(file_name))
                
    def load_from_str(s, data_root):
        assert len(s) >= 2
        assert s[0][0] == '{' and s[-1][-2:] == '}\n'
        s = ''.join(s)
        s = s.replace('\n', '')
        params = ast.literal_eval(s)
        if data_root:
            params.data_root = data_root
        return ExtractDic(params)
        
    def load(params_fn, data_root = None, verbose = 1):
        '''
        loads from file, adding '.param.txt' to name
        '''
        import ast
        with open(params_fn) as f:
            s = f.readlines()
            assert s[-1].startswith('hash:')
            params = ExtractDic.load_from_str(s[:-1], data_root)
        if verbose >= 2:
            print('params: '+ str(params) + '\nhash: ' + params.hash())
        if verbose >= 1:
            print('loaded from ' + str(params_fn))
        return params

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)