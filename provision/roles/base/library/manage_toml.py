# !/usr/bin/python

# Copyright: (c) 2018, Terry Jones <terry.jones@example.org>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)


ANSIBLE_METADATA = {
    'metadata_version': '1.1',
    'status': ['preview'],
    'supported_by': 'community'
}

DOCUMENTATION = '''
---
module: my_test

short_description: This is my test module

version_added: "2.4"

description:
    - "This is my longer description explaining my test module"

options:
    name:
        description:
            - This is the message to send to the test module
        required: true
    new:
        description:
            - Control to demo if the result of this module is changed or not
        required: false

extends_documentation_fragment:
    - azure

author:
    - Your Name (@yourhandle)
'''

EXAMPLES = '''
# Pass in a message
- name: Test with a message
  my_test:
    name: hello world

# pass in a message and have changed true
- name: Test with a message and changed output
  my_test:
    name: hello world
    new: true

# fail the module
- name: Test failure of the module
  my_test:
    name: fail me
'''

RETURN = '''
original_message:
    description: The original name param that was passed in
    type: str
    returned: always
message:
    description: The output message that the test module generates
    type: str
    returned: always
'''

from ansible.module_utils.basic import AnsibleModule
import tomlkit, tomlkit.items
from pathlib import Path
import os
from box import Box
import logging


def merge(a, b):
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(a, b) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in b.items():
        if isinstance(value, dict):
            # get node or create one
            node = a.setdefault(key, {})
            merge(node, value)
        else:
            a[key] = value

    return a


def tomlkit_to_popo(d):
    try:
        result = getattr(d, "value")
    except AttributeError:
        result = d

    if isinstance(result, list):
        result = [tomlkit_to_popo(x) for x in result]
    elif isinstance(result, dict):
        result = {
            tomlkit_to_popo(key): tomlkit_to_popo(val) for key, val in result.items()
        }
    elif isinstance(result, tomlkit.items.Integer):
        result = int(result)
    elif isinstance(result, tomlkit.items.Float):
        result = float(result)
    elif isinstance(result, tomlkit.items.String):
        result = str(result)
    elif isinstance(result, tomlkit.items.Bool):
        result = bool(result)

    return result


class TomlMan(object):
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.toml_data = tomlkit.loads(open(filepath).read())
        self.changed = False

    def get_popo_data(self):
        return tomlkit_to_popo(self.toml_data)

    def update_toml_data(self, user_data):
        original_toml_data = self.get_popo_data().copy()
        self.toml_data = merge(self.toml_data, user_data)
        if self.get_popo_data() != original_toml_data:
            self.changed = True
            with open(self.filepath, 'w') as file:
                file.write(tomlkit.dumps(self.toml_data))

    def copy_as(self, copy_info):
        method = copy_info.get('method', 'to_toml')
        if method:
            typename = method[3:]
            copy_path = copy_info.get('path', self.filepath.with_suffix(f'.{typename}'))
            if os.path.exists(copy_path):
                if Path(copy_path).suffix == self.filepath.suffix:
                    existing_file_content = TomlMan(copy_path).get_popo_data()
                else:
                    existing_file_content = getattr(Box, f'from_{typename}')(filename=copy_path).to_dict()
                if self.get_popo_data() == existing_file_content:
                    return
            self.changed = True
            getattr(Box(self.get_popo_data()), method)(filename=copy_path)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('qwe')


def run_module():
    module_args = dict(
        path=dict(type='str', required=True),
        data=dict(type='dict', required=False, default={}),
        copy_as=dict(type='dict', required=False, default={}),

    )

    module = AnsibleModule(argument_spec=module_args,
                           supports_check_mode=True, )
    tm = TomlMan(module.params['path'])

    copy_as = module.params.get('copy_as', {})
    data = module.params.get('data', {})
    result = module.params.copy()
    result.update(changed=False, out=tm.get_popo_data(), )
    for k in copy_as:
        if k not in ['path', 'method']:
            module.fail_json(msg=f'Undefined option "{k}" in "copy_as" parameter.',
                             **result)

    if module.check_mode: module.exit_json(**result)
    if data: tm.update_toml_data(data)
    if copy_as: tm.copy_as(copy_as)

    result['out'] = tm.get_popo_data()
    result['changed'] = result['changed'] or tm.changed
    module.exit_json(**result)


def main():
    run_module()
    # tm = TomlMan(r'/home/lgblkb/PycharmProjects/proj_1/pyproject.toml')
    # tm.update_toml_data(dict(qwe=1))
    # tm.copy_as(dict(data))


if __name__ == '__main__':
    main()
