

def set_params_lr_dict(model, base_lr, weight_decay, weight_dict, output=True):
    if output:  # output parms dict
        print('Set params dict lr!', weight_dict, 'Base:', base_lr, weight_decay)
    params_dict = weight_dict
    assert 'Default' not in params_dict.keys(), 'KEY \'Default\' should not in weight_dict(automantic set)'
    params_dict['Default'] = {}
    for name in params_dict.keys():
        params_dict[name]['params'] = []
    for name, param in model.named_parameters():
        result_key = 'Default'
        if not param.requires_grad:
            continue
        for key in weight_dict.keys():
            keys, keys_in_name = key.split(';'), True
            for k in keys:
                if k not in name:
                    keys_in_name = False
            if keys_in_name:
                result_key = key
                break
        if output:
            print('Set PARAM', name, 'USING KEY', result_key, [(key, value) for key, value in weight_dict[result_key].items() if key != 'params'])
        params_dict[result_key]['params'].append(param)
    parameters = []  # <<<  FOR Parameters
    for key, value in params_dict.items():
        value['Param_Name'] = key
        parameters.append(value)
    return parameters

