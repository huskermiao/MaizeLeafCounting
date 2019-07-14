import click


@click.command()
@click.argument('imgdir')
@click.argument('projid')
@click.option(
    '-s', '--subject',
    default=False,
    is_flag=True,
    help='Designate subject set id'
)
@click.option(
    '-q', '--quiet',
    default=False,
    is_flag=True,
    help='Silences output when uploading images to zooniverse'
)
def upload(imgdir, projid, subject, quiet):
    ''' Uploads images from the image directory to zooniverse project '''
    from zootils import upload as zoo_upload
    zoo_upload(imgdir, projid, subject, quiet)
    return


@click.command()
@click.argument('projid')
@click.argument('outfile')
@click.option(
    '-t', '--exp_type',
    default='classifications',
    help='Specify the type of export. Check Zooniverse help for available types'
)
@click.option(
    '-g', '--no_generate',
    default=False,
    is_flag=True,
    help='Generate a new export, versus download an existing one'
)
def export(projid, outfile, exp_type, no_generate):
    ''' Gets export from zooniverse project '''
    from zootils import export as zoo_export
    zoo_export(projid, outfile, exp_type, no_generate)
    return


@click.command()
@click.argument('imgdir')
def manifest(imgdir):
    ''' Generate a manifest in the directory specified by imgdir '''
    from zootils import manifest as zoo_manifest
    zoo_manifest(imgdir)
    return
