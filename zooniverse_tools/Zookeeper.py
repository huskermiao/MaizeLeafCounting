'''
NOTE:
    This utility is deprecated, please use the official Zooniverse project
    management cli tool Panoptes.

DESC:
    Schnablelab Zooniverse project management tool

DEPENDENCIES:
    click,
    panoptes-client
'''

import click


@click.group()
def Zookeeper():
    ''' Zooniverse project data management tools '''
    pass


from commands import (
    upload,
    export,
    manifest
)


Zookeeper.add_command(upload)
Zookeeper.add_command(export)
Zookeeper.add_command(manifest)


if __name__ == '__main__':
    Zookeeper()
