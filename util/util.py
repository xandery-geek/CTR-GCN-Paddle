import argparse


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])

    # return mod.comp1.comp2...
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)


def print_color(content, color='blue'):
    color_map = {
        "red": "31m",
        "green": "32m",
        "yellow": "33m",
        "blue": "34m",
    }
    print("\033[{}{}\033[0m".format(color_map[color], content))
