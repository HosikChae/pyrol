from __future__ import print_function
from builtins import range
from builtins import object
CLEAR = '\x1b[0m'

class COLOR(object):
    BLACK   = 0
    RED     = 1
    GREEN   = 2
    YELLOW  = 3
    BLUE    = 4
    MAGENTA = 5
    CYAN    = 6
    WHITE   = 7

class STYLE(object):
    DEFAULT         = 0
    BOLD            = 1
    FAINT           = 2
    ITALIC          = 3
    UNDERLINE       = 4
    BLINKING        = 5
    BLINKING_FAST   = 6
    REVERSE         = 7
    HIDE            = 8
    STRIKETHROUGH   = 9

def colorcode(f = COLOR.WHITE, b = COLOR.BLACK, s = STYLE.DEFAULT, end=False):
    code =  '\x1b[%d;%d;%dm' % (int(s), int(30+f), int(40+b))
    if end == True:
        return code, CLEAR
    else:
        return code

def colored_string(str, f = COLOR.WHITE, b = COLOR.BLACK, s = STYLE.DEFAULT):
    return "%s%s%s"%(colorcode(f=f, b=b, s=s), str, CLEAR)

def print_color_table():
    style_list = ["Default", "Bold", "Faint", "Italic", "Underline", "Blinking", "Blinking Fast", "Reverse", "Hide", "Strikethough"]
    color_list = ["K", "R", "G", "Y", "B", "M", "C", "W"]
    for sdx in range(10):
        print("Style: %s" % (style_list[sdx]))
        for fdx in range(8):
            print("%s  "%(color_list[fdx]), end="")
            for bdx in range(8):
                code = colorcode(f=fdx, b=bdx, s=sdx)
                print("%s %d;%d;%d %s"%(code, sdx, 30+fdx, 40+bdx ,CLEAR), end="")

            print("")

        print("")
        print("")

kHEADER = '\033[95m'
kGPD_X = kOK_BLUE = '\033[94m'
kGPD_Y = kOK_GREEN = '\033[92m'
kGPD_A = kWARNING_YELLOW = '\033[93m'
kGPD_B = kFAIL_RED = '\033[91m'
kENDC = '\033[0m'
# BOLD = '\033[1m'
# UNDERLINE = '\033[4m'

if __name__ == "__main__":
    print_color_table()