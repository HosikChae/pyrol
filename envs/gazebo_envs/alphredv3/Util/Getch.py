#!usr/bin/env python
from __future__ import print_function
__author__ = "Min Sung Ahn"
__email__ = "aminsung@gmail.com"
__copyright__ = "Copyright 2017 RoMeLa"
__date__ = "November 5, 2017"

__version__ = "0.0.1"
__status__ = "Prototype"

'''

'''

import pdb
import sys
import tty
import termios

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

import select
old_settings = termios.tcgetattr(sys.stdin)
def getch_nb(timeout=0.5): # non-blocking version
    try:
        tty.setcbreak(sys.stdin.fileno())
        if select.select([sys.stdin], [], [], timeout) == ([sys.stdin], [], []):
            c = sys.stdin.read(1)
        else:
            c = None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return c

# import curses, time
# def getch_n(message=""):
#     try:
#         win = curses.initscr()
#         # win.addstr(0, 0, message)
#         while True:
#             ch = win.getch()
#             if ch in range(32, 127): break
#             time.sleep(0.05)
#     except: raise
#     finally:
#         curses.endwin()
#         return chr(ch)

if __name__ ==  '__main__':
    while True:
        cmd = getch_nb()
        if cmd == None:
            continue
        print ("The character you typed is: {}".format(cmd))
        if cmd == '\x1b':  # ESC
            break

    # while True:
    #     cmd = getch()
    #     print ("The character you typed is: {}".format(cmd))
    #
    # print("Done!")