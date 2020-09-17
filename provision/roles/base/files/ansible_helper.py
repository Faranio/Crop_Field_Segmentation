import glob
import os
import toml
import sys
import logging
import subprocess
from pathlib import Path
from subprocess import call

import click
from box import Box

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('lgblkb')


def run_cmd_parts(parts):
    cmd = " ".join(map(str, parts))
    logger.debug("cmd: %s", cmd)
    call(parts)


context_settings = dict(ignore_unknown_options=True, )
this_folder = Path(__file__).parent
provision_folder = this_folder.parents[2]
vault_parts = ['--vault-password-file', this_folder.joinpath('.secret')]


@click.group()
def main():
    pass


# region Encrypt/Decrypt:


@main.command(context_settings=context_settings)
@click.option("-p", "--provisional", is_flag=True)
@click.option("-i", "--inventory")
@click.argument('filename')
@click.argument('other_args', nargs=-1, type=click.UNPROCESSED)
def encrypt(provisional, inventory, filename, other_args):
    encrypt_decrypt('encrypt', provisional, inventory, filename, other_args)


@main.command(context_settings=context_settings)
@click.option("-p", "--provisional", is_flag=True)
@click.option("-i", "--inventory")
@click.argument('filename')
@click.argument('other_args', nargs=-1, type=click.UNPROCESSED)
def decrypt(provisional, inventory, filename, other_args):
    encrypt_decrypt('decrypt', provisional, inventory, filename, other_args)


def encrypt_decrypt(action, *args):
    provisional, inventory, filename, other_args = args
    if provisional and inventory:
        raise ValueError('Either provisional or inventory filename should be specified. Not both.')
    elif provisional or inventory:
        if provisional:
            filename = provision_folder.joinpath(filename)
        elif inventory:
            filename = provision_folder.joinpath('envs', inventory, filename)
        if not filename.suffix:
            filename = filename.with_suffix('.yaml')
            encrypt_decrypt_file(action, filename, other_args)
    else:
        if str(filename).lower() == 'all':
            glob_patterns = Box(toml.loads(open(os.path.abspath('pyproject.toml')).read())).tool.project.encrypted
            for glob_pattern in glob_patterns:
                filepaths = glob.glob(glob_pattern, recursive=True)
                for filepath in filepaths:
                    encrypt_decrypt_file(action, filepath, other_args)
        else:
            encrypt_decrypt_file(action, filename, other_args)


def encrypt_decrypt_file(action, file, other_args):
    if not os.path.exists(file): raise FileNotFoundError(file)
    parts = ['ansible-vault', action, *vault_parts, file, *other_args]
    run_cmd_parts(parts)
    return


# endregion

@main.command(context_settings=context_settings)
@click.argument('playbook')
@click.option("-i", "--inventory", default="development", show_default=True)
@click.option('--vault/--no-vault', default=True, show_default=True)
@click.argument('other_args', nargs=-1, type=click.UNPROCESSED)
def play(playbook, inventory, vault, other_args):
    parts = ['ansible-playbook']
    if vault: parts.extend(vault_parts)
    parts.extend(['--inventory', provision_folder.joinpath('envs', inventory)])
    parts.extend([provision_folder.joinpath(playbook + '.yaml'), *other_args])
    run_cmd_parts(parts)


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def execute2(command):
    subprocess.check_call(command, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == '__main__':
    main()
