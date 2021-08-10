from __future__ import annotations

import argparse
from ipaddress import ip_address

description = 'Berkeley Quantum Synthesis Toolkit CLI'
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    '--ip',
    action='store',
    type=ip_address,
    help='IP Address of main node.',
    default=ip_address('0.0.0.0'),
)

parser.add_argument(
    '--port',
    action='store',
    type=int,
    help='The port the main node runs on.',
    default=35536,
)
