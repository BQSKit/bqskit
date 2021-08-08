from __future__ import annotations

import argparse
from array import array
from ipaddress import ip_address
from multiprocessing.connection import Client
from multiprocessing.connection import Listener

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

args = parser.parse_args()
address = (str(args.ip), args.port)

if str(args.ip) == '0.0.0.0':
    with Listener(address, authkey=b'secret password') as listener:
        with listener.accept() as conn:
            print('connection accepted from', listener.last_accepted)

            conn.send([2.25, None, 'junk', float])

            conn.send_bytes(b'hello')

            conn.send_bytes(array('i', [42, 1729]))
else:
    with Client(address, authkey=b'secret password') as conn:
        print(conn.recv())                  # => [2.25, None, 'junk', float]

        print(conn.recv_bytes())            # => 'hello'

        arr = array('i', [0, 0, 0, 0, 0])
        print(conn.recv_bytes_into(arr))    # => 8
        print(arr)                          # => array('i', [42, 1729, 0, 0, 0])
