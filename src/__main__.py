import click
from timemachine import timemachine
from generator import generator
from grapher import grapher

@click.group()
def cli():
    pass

cli.add_command(timemachine)
cli.add_command(generator)
cli.add_command(grapher)

if __name__ == "__main__":
    cli()
